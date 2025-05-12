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

def create_team_stats(df):
    """팀 통계 관련 특성 생성 함수"""
    team_stats = []
    
    for team in df['팀명'].unique():
        team_data = df[df['팀명'] == team].sort_values('날짜')
        
        # 이동평균 통계
        for window in [3, 5, 10]:
            team_data[f'최근{window}경기_승률'] = (
                team_data['승리여부'].shift(1).eq('승')
                .rolling(window, min_periods=1)
                .mean()
                .fillna(0.5)
            )
            
            team_data[f'최근{window}경기_득점'] = (
                team_data['득점'].shift(1)
                .rolling(window, min_periods=1)
                .mean()
                .fillna(team_data['득점'].shift(1).mean())
            )
        
        team_stats.append(team_data)
    
    return pd.concat(team_stats)

def create_matchup_features(df):
    """팀 간 상대 전적 특성 생성 함수"""
    df = df.copy()
    
    # 상대팀 정보 추가
    df['상대팀'] = df.groupby('경기ID')['팀명'].transform(
        lambda x: x.iloc[::-1].values if len(x) == 2 else None
    )
    
    matchup_stats = []
    
    for team in df['팀명'].unique():
        team_data = df[df['팀명'] == team].sort_values('날짜')
        
        for opponent in df['팀명'].unique():
            if team != opponent:
                mask = (team_data['상대팀'] == opponent)
                team_data.loc[mask, f'VS_{opponent}_승률'] = (
                    team_data.loc[mask, '승리여부'].shift(1).eq('승')
                    .expanding()
                    .mean()
                    .fillna(0.5)
                )
        
        matchup_stats.append(team_data)
    
    return pd.concat(matchup_stats)

def create_pitching_features(df):
    """투수 관련 특성 생성 함수"""
    df = df.copy()
    
    # 상대팀 득점을 실점으로 사용
    df['실점'] = df.groupby('경기ID')['득점'].transform('sum') - df['득점']
    
    pitching_stats = []
    
    for team in df['팀명'].unique():
        team_data = df[df['팀명'] == team].sort_values('날짜')
        
        # 이동평균 실점
        team_data['최근실점'] = (
            team_data['실점'].shift(1)
            .rolling(10, min_periods=1)
            .mean()
            .fillna(team_data['실점'].shift(1).mean())
        )
        
        # 실점 추세
        team_data['실점추세'] = (
            team_data['실점'].shift(1)
            .rolling(5, min_periods=1)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            .fillna(0)
        )
        
        pitching_stats.append(team_data)
    
    return pd.concat(pitching_stats)

def preprocess_data(df):
    """데이터 전처리를 수행하는 함수"""
    logging.info("데이터 전처리 시작...")
    
    # 날짜 변환
    df['날짜'] = pd.to_datetime(df['날짜'])
    
    # 특성 생성
    df = create_basic_features(df)
    df = create_time_features(df)
    df = create_team_stats(df)
    df = create_matchup_features(df)
    df = create_pitching_features(df)
    
    # 범주형 변수 인코딩
    le = LabelEncoder()
    categorical_columns = ['팀명', '경기장', '홈/원정']
    
    for col in categorical_columns:
        df[col + '_인코딩'] = le.fit_transform(df[col])
    
    return df

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

def main():
    try:
        # 1. 데이터 로드
        df = load_data()
        
        # 2. 데이터 전처리
        df = preprocess_data(df)
        
        # 3. 특성 선택
        features = [col for col in df.columns if any(x in col for x in [
            '인코딩', '최근', 'VS_', '실점', '추세', '율', '경기장_평균'
        ])]
        
        # 4. 데이터 분할
        X = df[features]
        y = (df['승리여부'] == '승').astype(int)
        
        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(X))
        train_index, test_index = splits[-1]
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # 5. 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 6. 기본 모델 생성
        base_models = create_models()
        
        # 7. 앙상블 모델 생성 및 학습
        voting_clf = create_voting_classifier(base_models)
        voting_clf.fit(X_train_scaled, y_train)
        
        # 8. 예측
        y_pred_proba = voting_clf.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # 9. 성능 평가
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        logging.info(f'\n고급 앙상블 모델 정확도: {accuracy:.4f}')
        logging.info(f'고급 앙상블 모델 AUC: {auc:.4f}')
        
        # 10. 개별 모델 성능 평가
        for name, model in base_models.items():
            model.fit(X_train_scaled, y_train)
            model_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            model_pred = (model_pred_proba > 0.5).astype(int)
            
            model_accuracy = accuracy_score(y_test, model_pred)
            model_auc = roc_auc_score(y_test, model_pred_proba)
            
            logging.info(f'\n{name} 모델 정확도: {model_accuracy:.4f}')
            logging.info(f'{name} 모델 AUC: {model_auc:.4f}')
        
        # 11. 결과 저장
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