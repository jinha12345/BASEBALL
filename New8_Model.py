import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import traceback

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_win_loss(df):
    """승리여부를 계산하는 함수"""
    df = df.copy()
    df['경기ID'] = df.groupby(['날짜', '경기장']).ngroup()
    
    # 각 경기별로 승리팀 계산
    game_results = []
    for game_id in df['경기ID'].unique():
        game = df[df['경기ID'] == game_id].copy()
        if len(game) == 2:  # 두 팀의 데이터가 모두 있는 경우만
            game = game.sort_values('득점', ascending=False)
            game.iloc[0, game.columns.get_loc('승리여부')] = '승'
            game.iloc[1, game.columns.get_loc('승리여부')] = '패'
            game_results.append(game)
    
    return pd.concat(game_results)

def load_data(file_path='BASEBALL_stats_15.xlsx'):
    """데이터를 로드하고 기본 정보를 출력하는 함수"""
    logging.info("데이터 로드 시작...")
    try:
        df = pd.read_excel(file_path)
        df['승리여부'] = ''  # 승리여부 컬럼 추가
        df = calculate_win_loss(df)  # 승리여부 계산
        logging.info(f"데이터 로드 완료. 데이터 크기: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"데이터 로드 중 오류 발생: {str(e)}")
        raise

def create_basic_features(df):
    """기본 특성 생성 함수"""
    # 루타 계산
    df['루타'] = df['안타'] + df['2루타'] + 2*df['3루타'] + 3*df['홈런']
    
    # 팀 성적 관련 특성
    df['득점률'] = df['득점'] / df['타수']
    df['장타율'] = df['루타'] / df['타수']
    df['출루율'] = (df['안타'] + df['볼넷'] + df['사구']) / (df['타수'] + df['볼넷'] + df['사구'])
    
    # 경기장 특성
    stadium_stats = df.groupby('경기장').agg({
        '득점': 'mean',
        '안타': 'mean',
        '홈런': 'mean'
    }).reset_index()
    
    stadium_stats.columns = ['경기장', '경기장_평균득점', '경기장_평균안타', '경기장_평균홈런']
    df = df.merge(stadium_stats, on='경기장', how='left')
    
    return df

def create_time_features(df):
    """시간 관련 특성 생성 함수"""
    df['요일'] = pd.to_datetime(df['날짜']).dt.dayofweek
    df['월'] = pd.to_datetime(df['날짜']).dt.month
    df['주말'] = df['요일'].isin([5, 6]).astype(int)
    
    df['시즌진행도'] = df.groupby('연도')['날짜'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    
    return df

def create_team_stats(df, target_date):
    """팀 통계 관련 특성 생성 함수"""
    team_stats = []
    
    for team in df['팀명'].unique():
        # 해당 날짜 이전의 데이터만 선택
        team_data = df[
            (df['팀명'] == team) & 
            (df['날짜'] < target_date)
        ].sort_values('날짜')
        
        if len(team_data) < 1:
            continue
        
        # 최근 통계 계산
        recent_stats = pd.DataFrame(index=[0])
        recent_stats['팀명'] = team
        recent_stats['날짜'] = target_date
        
        # 이동평균 통계
        for window in [3, 5, 10]:
            recent_stats[f'최근{window}경기_승률'] = (
                team_data['승리여부'].eq('승')
                .rolling(window, min_periods=1)
                .mean()
                .iloc[-1]
            )
            
            recent_stats[f'최근{window}경기_득점'] = (
                team_data['득점']
                .rolling(window, min_periods=1)
                .mean()
                .iloc[-1]
            )
        
        team_stats.append(recent_stats)
    
    if not team_stats:
        return pd.DataFrame()
    
    return pd.concat(team_stats, ignore_index=True)

def create_matchup_features(df, target_date):
    """팀 간 상대 전적 특성 생성 함수"""
    df = df.copy()
    
    # 상대팀 정보 추가
    df['상대팀'] = df.groupby('경기ID')['팀명'].transform(
        lambda x: x.iloc[::-1].values if len(x) == 2 else None
    )
    
    matchup_stats = []
    
    for team in df['팀명'].unique():
        # 해당 날짜 이전의 데이터만 선택
        team_data = df[
            (df['팀명'] == team) & 
            (df['날짜'] < target_date)
        ].sort_values('날짜')
        
        if len(team_data) < 1:
            continue
        
        recent_stats = pd.DataFrame(index=[0])
        recent_stats['팀명'] = team
        recent_stats['날짜'] = target_date
        
        for opponent in df['팀명'].unique():
            if team != opponent:
                mask = (team_data['상대팀'] == opponent)
                if mask.any():
                    recent_stats[f'VS_{opponent}_승률'] = (
                        team_data.loc[mask, '승리여부'].eq('승').mean()
                        if len(team_data.loc[mask]) > 0
                        else 0.5
                    )
                else:
                    recent_stats[f'VS_{opponent}_승률'] = 0.5
        
        matchup_stats.append(recent_stats)
    
    if not matchup_stats:
        return pd.DataFrame()
    
    return pd.concat(matchup_stats, ignore_index=True)

def create_pitching_features(df, target_date):
    """투수 관련 특성 생성 함수"""
    df = df.copy()
    
    # 상대팀 득점을 실점으로 사용
    df['실점'] = df.groupby('경기ID')['득점'].transform('sum') - df['득점']
    
    pitching_stats = []
    
    for team in df['팀명'].unique():
        # 해당 날짜 이전의 데이터만 선택
        team_data = df[
            (df['팀명'] == team) & 
            (df['날짜'] < target_date)
        ].sort_values('날짜')
        
        if len(team_data) < 1:
            continue
        
        recent_stats = pd.DataFrame(index=[0])
        recent_stats['팀명'] = team
        recent_stats['날짜'] = target_date
        
        # 이동평균 실점
        recent_stats['최근실점'] = (
            team_data['실점']
            .rolling(10, min_periods=1)
            .mean()
            .iloc[-1]
        )
        
        # 실점 추세
        if len(team_data) >= 5:
            recent_data = team_data.tail(5)
            recent_stats['실점추세'] = np.polyfit(
                range(len(recent_data)), 
                recent_data['실점'].values, 
                1
            )[0]
        else:
            recent_stats['실점추세'] = 0
        
        pitching_stats.append(recent_stats)
    
    if not pitching_stats:
        return pd.DataFrame()
    
    return pd.concat(pitching_stats, ignore_index=True)

def preprocess_data(df):
    """데이터 전처리를 수행하는 함수"""
    logging.info("데이터 전처리 시작...")
    
    # 날짜 변환
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.sort_values('날짜')
    
    # 각 날짜별로 특성 생성
    processed_data = []
    for date in df['날짜'].unique():
        # 해당 날짜의 경기 데이터
        current_games = df[df['날짜'] == date].copy()
        
        # 기본 특성 생성
        current_games = create_basic_features(current_games)
        current_games = create_time_features(current_games)
        
        # 시계열 특성 생성 (해당 날짜 이전 데이터만 사용)
        team_stats = create_team_stats(df, date)
        matchup_stats = create_matchup_features(df, date)
        pitching_stats = create_pitching_features(df, date)
        
        # 현재 경기 데이터와 통계 병합
        if not team_stats.empty:
            current_games = current_games.merge(
                team_stats, on=['팀명', '날짜'], how='left')
        if not matchup_stats.empty:
            current_games = current_games.merge(
                matchup_stats, on=['팀명', '날짜'], how='left')
        if not pitching_stats.empty:
            current_games = current_games.merge(
                pitching_stats, on=['팀명', '날짜'], how='left')
        
        processed_data.append(current_games)
    
    # 모든 처리된 데이터 합치기
    df_processed = pd.concat(processed_data, ignore_index=True)
    
    # 범주형 변수 인코딩
    le = LabelEncoder()
    categorical_columns = ['팀명', '경기장', '홈/원정']
    
    for col in categorical_columns:
        df_processed[col + '_인코딩'] = le.fit_transform(df_processed[col])
    
    return df_processed

def create_models():
    """기본 모델들을 생성하는 함수"""
    models = {
        'xgb': xgb.XGBClassifier(
            learning_rate=0.02,
            max_depth=4,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        'lgb': lgb.LGBMClassifier(
            learning_rate=0.02,
            max_depth=4,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        'rf': RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_split=5,
            random_state=42
        )
    }
    return models

def create_voting_classifier(models):
    """보팅 분류기를 생성하는 함수"""
    estimators = [(name, model) for name, model in models.items()]
    return VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=[0.4, 0.4, 0.2]  # XGBoost와 LightGBM에 더 높은 가중치
    )

def create_stacking_classifier(models):
    """스태킹 분류기를 생성하는 함수"""
    estimators = [(name, model) for name, model in models.items()]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=3,
        n_jobs=-1
    )

def optimize_weights(models, X_val, y_val):
    """검증 세트를 사용하여 최적의 가중치를 찾는 함수"""
    logging.info("\n앙상블 가중치 최적화 중...")
    
    # 각 모델의 예측 확률
    model_probs = []
    for name, model in models.items():
        probs = model.predict_proba(X_val)[:, 1]
        model_probs.append(probs)
    
    # 그리드 서치로 최적 가중치 탐색
    best_score = 0
    best_weights = None
    
    for w1 in range(1, 10):
        for w2 in range(1, 10):
            for w3 in range(1, 10):
                weights = np.array([w1, w2, w3]) / (w1 + w2 + w3)
                
                # 가중 평균 예측
                ensemble_probs = np.zeros_like(model_probs[0])
                for w, probs in zip(weights, model_probs):
                    ensemble_probs += w * probs
                
                # 성능 평가
                score = roc_auc_score(y_val, ensemble_probs)
                
                if score > best_score:
                    best_score = score
                    best_weights = weights
    
    logging.info(f"최적 가중치: {dict(zip(models.keys(), best_weights))}")
    logging.info(f"최적 AUC 점수: {best_score:.4f}")
    
    return best_weights

def create_weighted_voting_classifier(models, weights):
    """가중치가 적용된 투표 분류기를 생성하는 함수"""
    return VotingClassifier(
        estimators=list(models.items()),
        voting='soft',
        weights=weights
    )

def main():
    try:
        # 1. 데이터 로드
        df = load_data()
        
        # 2. 데이터 전처리
        df = preprocess_data(df)
        
        # 3. 특성 선택
        features = [
            '팀명_인코딩', '경기장_인코딩', '홈/원정_인코딩',
            '최근승률', '최근평균득점', '최근평균안타', '최근평균홈런',
            '최근평균타율', '최근평균출루율', '득점률', '출루율'
        ]
        
        # 4. 데이터 분할 (시계열 고려)
        X = df[features]
        y = (df['승리여부'] == '승').astype(int)
        
        # 훈련, 검증, 테스트 세트 분할
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)
        
        X_train = X[:train_size]
        X_val = X[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        
        y_train = y[:train_size]
        y_val = y[train_size:train_size+val_size]
        y_test = y[train_size+val_size:]
        
        # 5. 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 6. 기본 모델 생성 및 학습
        models = create_models()
        
        logging.info("\n기본 모델 학습 중...")
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            val_pred = model.predict(X_val_scaled)
            val_accuracy = accuracy_score(y_val, val_pred)
            logging.info(f"{name} 검증 정확도: {val_accuracy:.4f}")
        
        # 7. 최적 가중치 탐색
        best_weights = optimize_weights(models, X_val_scaled, y_val)
        
        # 8. 가중치 투표 앙상블 생성 및 학습
        weighted_ensemble = create_weighted_voting_classifier(models, best_weights)
        weighted_ensemble.fit(X_train_scaled, y_train)
        
        # 9. 예측 및 평가
        y_pred_proba = weighted_ensemble.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        logging.info(f'\n가중치 앙상블 모델 정확도: {accuracy:.4f}')
        logging.info(f'가중치 앙상블 모델 AUC: {auc:.4f}')
        
        # 10. 결과 저장
        results = pd.DataFrame({
            '실제값': y_test,
            '예측값': y_pred,
            '예측확률': y_pred_proba
        })
        results.to_csv('prediction_results_ensemble.csv', index=False, encoding='utf-8-sig')
        
        logging.info("\n모든 과정이 완료되었습니다.")
        
    except Exception as e:
        logging.error(f"\n오류 발생: {str(e)}")
        logging.error("\n상세 오류 정보:")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 