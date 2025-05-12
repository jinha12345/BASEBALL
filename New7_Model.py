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

def calculate_team_pitching_stats(df):
    """팀별 투수 성적 통계를 계산하는 함수"""
    pitching_stats = []
    
    for team in df['팀명'].unique():
        team_data = df[df['팀명'] == team].sort_values('날짜')
        
        # 상대팀 득점을 투수 실점으로 사용
        team_data['실점'] = team_data.groupby('경기ID')['득점'].transform('sum') - team_data['득점']
        
        # 이닝당 실점 (ERA와 유사)
        team_data['이닝당실점'] = team_data['실점'].rolling(10, min_periods=1).mean()
        
        # 투수 안정성 (실점 표준편차)
        team_data['실점안정성'] = team_data['실점'].rolling(10, min_periods=1).std()
        
        # 최근 10경기 평균 실점
        team_data['최근실점'] = team_data['실점'].rolling(10, min_periods=1).mean()
        
        # 홈/원정 구분 실점
        for location in ['홈', '원정']:
            mask = team_data['홈/원정'] == location
            team_data.loc[mask, f'{location}_실점'] = (
                team_data.loc[mask, '실점']
                .rolling(10, min_periods=1)
                .mean()
            )
        
        pitching_stats.append(team_data)
    
    return pd.concat(pitching_stats)

def calculate_matchup_stats(df):
    """팀 간 상대 전적 및 투수 성적을 계산하는 함수"""
    df = df.copy()
    
    # 상대팀 정보 추가
    df['상대팀'] = df.groupby('경기ID')['팀명'].transform(
        lambda x: x.iloc[::-1].values if len(x) == 2 else None
    )
    
    matchup_stats = []
    
    for team in df['팀명'].unique():
        team_data = df[df['팀명'] == team].sort_values('날짜')
        
        # 상대팀별 투수 성적
        for opponent in df['팀명'].unique():
            if team != opponent:
                mask = (team_data['상대팀'] == opponent)
                
                # 상대팀 상대 평균 실점
                team_data.loc[mask, f'VS_{opponent}_실점'] = (
                    team_data.loc[mask, '실점']
                    .expanding()
                    .mean()
                    .fillna(team_data['실점'].mean())
                )
                
                # 상대팀 상대 승률
                team_data.loc[mask, f'VS_{opponent}_승률'] = (
                    team_data.loc[mask, '승리여부'].eq('승')
                    .expanding()
                    .mean()
                    .fillna(0.5)
                )
        
        matchup_stats.append(team_data)
    
    return pd.concat(matchup_stats)

def create_advanced_pitching_features(df):
    """고급 투수 관련 특성 생성 함수"""
    # 실점 추세 (기울기)
    df['실점추세'] = (
        df.groupby('팀명')['실점']
        .rolling(5, min_periods=1)
        .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
        .reset_index(level=0, drop=True)
    )
    
    # 실점 변동성
    df['실점변동성'] = (
        df.groupby('팀명')['실점']
        .rolling(10, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )
    
    # 연속 퀄리티스타트 (6이닝 3실점 이하로 가정)
    df['퀄리티스타트'] = df['실점'] <= 3
    df['연속QS'] = (
        df.groupby('팀명')['퀄리티스타트']
        .rolling(5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    
    return df

def preprocess_data(df):
    """데이터 전처리를 수행하는 함수"""
    logging.info("데이터 전처리 시작...")
    
    # 날짜 변환
    df['날짜'] = pd.to_datetime(df['날짜'])
    
    # 투수 통계 계산
    df = calculate_team_pitching_stats(df)
    df = calculate_matchup_stats(df)
    df = create_advanced_pitching_features(df)
    
    # 범주형 변수 인코딩
    le = LabelEncoder()
    categorical_columns = ['팀명', '경기장', '홈/원정']
    
    for col in categorical_columns:
        df[col + '_인코딩'] = le.fit_transform(df[col])
    
    return df

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