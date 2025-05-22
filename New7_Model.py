import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
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

def calculate_team_pitching_stats(df, target_date):
    """팀별 투수 성적 통계를 계산하는 함수"""
    pitching_stats = []
    
    for team in df['팀명'].unique():
        # 해당 날짜 이전의 데이터만 선택
        team_data = df[
            (df['팀명'] == team) & 
            (df['날짜'] < target_date)
        ].sort_values('날짜')
        
        if len(team_data) < 1:
            continue
        
        # 상대팀 득점을 투수 실점으로 사용
        team_data['실점'] = team_data.groupby('경기ID')['득점'].transform('sum') - team_data['득점']
        
        # 이닝 정보 추가 (9이닝 기준)
        team_data['투구이닝'] = 9  # 기본값
        team_data.loc[team_data['홈/원정'] == '홈', '투구이닝'] = 8  # 9회말 공격팀이 이기면 8이닝
        
        # 최근 통계 계산
        recent_stats = pd.DataFrame({
            '이닝당실점': (
                team_data['실점'].rolling(10, min_periods=1).sum() /
                team_data['투구이닝'].rolling(10, min_periods=1).sum()
            ).iloc[-1] * 9,  # 9이닝 기준으로 환산
            '실점안정성': team_data['실점'].rolling(10, min_periods=1).std().iloc[-1],
            '최근실점': team_data['실점'].rolling(10, min_periods=1).mean().iloc[-1],
            '최근완투율': (
                team_data['투구이닝'].eq(9)
                .rolling(10, min_periods=1)
                .mean()
                .iloc[-1]
            ),
            '최근QS비율': (
                (team_data['투구이닝'] >= 6) & (team_data['실점'] <= 3)
            ).rolling(10, min_periods=1).mean().iloc[-1]
        }, index=[0])
        
        # 홈/원정 구분 실점
        for location in ['홈', '원정']:
            mask = team_data['홈/원정'] == location
            if mask.any():
                recent_stats[f'{location}_실점'] = (
                    team_data.loc[mask, '실점']
                    .rolling(10, min_periods=1)
                    .mean()
                    .iloc[-1] if len(team_data.loc[mask]) > 0 else np.nan
                )
                
                recent_stats[f'{location}_이닝당실점'] = (
                    (team_data.loc[mask, '실점'].rolling(10, min_periods=1).sum() /
                     team_data.loc[mask, '투구이닝'].rolling(10, min_periods=1).sum() * 9)
                    .iloc[-1] if len(team_data.loc[mask]) > 0 else np.nan
                )
            else:
                recent_stats[f'{location}_실점'] = np.nan
                recent_stats[f'{location}_이닝당실점'] = np.nan
        
        recent_stats['팀명'] = team
        recent_stats['날짜'] = target_date
        pitching_stats.append(recent_stats)
    
    if not pitching_stats:
        return pd.DataFrame()
    
    return pd.concat(pitching_stats, ignore_index=True)

def calculate_matchup_stats(df, target_date):
    """팀 간 상대 전적 및 투수 성적을 계산하는 함수"""
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
        
        # 상대팀별 투수 성적
        recent_stats = pd.DataFrame(index=[0])
        recent_stats['팀명'] = team
        recent_stats['날짜'] = target_date
        
        for opponent in df['팀명'].unique():
            if team != opponent:
                mask = (team_data['상대팀'] == opponent)
                if mask.any():
                    # 상대팀 상대 평균 실점
                    recent_stats[f'VS_{opponent}_실점'] = (
                        team_data.loc[mask, '실점'].mean()
                        if len(team_data.loc[mask]) > 0
                        else team_data['실점'].mean()
                    )
                    
                    # 상대팀 상대 승률
                    recent_stats[f'VS_{opponent}_승률'] = (
                        team_data.loc[mask, '승리여부'].eq('승').mean()
                        if len(team_data.loc[mask]) > 0
                        else 0.5
                    )
                else:
                    recent_stats[f'VS_{opponent}_실점'] = team_data['실점'].mean()
                    recent_stats[f'VS_{opponent}_승률'] = 0.5
        
        matchup_stats.append(recent_stats)
    
    if not matchup_stats:
        return pd.DataFrame()
    
    return pd.concat(matchup_stats, ignore_index=True)

def create_advanced_pitching_features(df, target_date):
    """고급 투수 관련 특성 생성 함수"""
    advanced_stats = []
    
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
        
        # 실점 추세 (기울기)
        if len(team_data) >= 5:
            recent_data = team_data.tail(5)
            recent_stats['실점추세'] = np.polyfit(
                range(len(recent_data)), 
                recent_data['실점'].values, 
                1
            )[0]
        else:
            recent_stats['실점추세'] = 0
        
        # 실점 변동성
        recent_stats['실점변동성'] = (
            team_data['실점']
            .rolling(10, min_periods=1)
            .std()
            .iloc[-1]
        )
        
        # 이닝당실점 추세
        team_data['이닝당실점'] = team_data['실점'] / team_data['투구이닝'] * 9
        if len(team_data) >= 5:
            recent_data = team_data.tail(5)
            recent_stats['이닝당실점추세'] = np.polyfit(
                range(len(recent_data)),
                recent_data['이닝당실점'].values,
                1
            )[0]
        else:
            recent_stats['이닝당실점추세'] = 0
        
        # 연속 퀄리티스타트
        team_data['퀄리티스타트'] = (team_data['투구이닝'] >= 6) & (team_data['실점'] <= 3)
        recent_stats['연속QS'] = (
            team_data['퀄리티스타트']
            .rolling(5, min_periods=1)
            .sum()
            .iloc[-1]
        )
        
        # 최근 완투 횟수
        team_data['완투'] = team_data['투구이닝'] == 9
        recent_stats['최근완투'] = (
            team_data['완투']
            .rolling(10, min_periods=1)
            .sum()
            .iloc[-1]
        )
        
        advanced_stats.append(recent_stats)
    
    if not advanced_stats:
        return pd.DataFrame()
    
    return pd.concat(advanced_stats, ignore_index=True)

def preprocess_data(df):
    """데이터 전처리를 수행하는 함수"""
    logging.info("데이터 전처리 시작...")
    
    # 날짜 변환
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.sort_values('날짜')
    
    # 각 날짜별로 통계 계산
    processed_data = []
    for date in df['날짜'].unique():
        # 해당 날짜의 경기 데이터
        current_games = df[df['날짜'] == date].copy()
        
        # 투수 통계 계산 (해당 날짜 이전 데이터만 사용)
        pitching_stats = calculate_team_pitching_stats(df, date)
        matchup_stats = calculate_matchup_stats(df, date)
        advanced_stats = create_advanced_pitching_features(df, date)
        
        # 현재 경기 데이터와 통계 병합
        if not pitching_stats.empty:
            current_games = current_games.merge(
                pitching_stats, on=['팀명', '날짜'], how='left')
        if not matchup_stats.empty:
            current_games = current_games.merge(
                matchup_stats, on=['팀명', '날짜'], how='left')
        if not advanced_stats.empty:
            current_games = current_games.merge(
                advanced_stats, on=['팀명', '날짜'], how='left')
        
        processed_data.append(current_games)
    
    # 모든 처리된 데이터 합치기
    df_processed = pd.concat(processed_data, ignore_index=True)
    
    # 범주형 변수 인코딩
    le = LabelEncoder()
    categorical_columns = ['팀명', '경기장', '홈/원정']
    
    for col in categorical_columns:
        df_processed[col + '_인코딩'] = le.fit_transform(df_processed[col])
    
    return df_processed

def main():
    try:
        # 1. 데이터 로드
        df = load_data()
        
        # 2. 데이터 전처리
        df = preprocess_data(df)
        
        # 3. 특성 선택
        features = [col for col in df.columns if any(x in col for x in [
            '인코딩', '실점', 'VS_', '연속QS', '추세', '변동성'
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
        
        # 6. 모델 학습
        model = xgb.XGBClassifier(
            learning_rate=0.02,
            max_depth=4,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # 7. 예측
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # 8. 성능 평가
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        logging.info(f'\n투수 통계 기반 모델 정확도: {accuracy:.4f}')
        logging.info(f'투수 통계 기반 모델 AUC: {auc:.4f}')
        
        # 9. 특성 중요도 시각화
        plt.figure(figsize=(12, 6))
        importance = pd.Series(model.feature_importances_, index=features)
        importance.sort_values().plot(kind='barh')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance_pitching.png')
        plt.close()
        
        # 10. 결과 저장
        results = pd.DataFrame({
            '실제값': y_test,
            '예측값': y_pred,
            '예측확률': y_pred_proba
        })
        results.to_csv('prediction_results_pitching.csv', index=False, encoding='utf-8-sig')
        
        logging.info("\n모든 과정이 완료되었습니다.")
        
    except Exception as e:
        logging.error(f"\n오류 발생: {str(e)}")
        logging.error("\n상세 오류 정보:")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 