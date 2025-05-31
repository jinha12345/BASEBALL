import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import traceback
from datetime import timedelta

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

def create_time_features(df):
    """시간 관련 특성 생성 함수"""
    # 기본 시간 특성
    df['요일'] = pd.to_datetime(df['날짜']).dt.dayofweek
    df['월'] = pd.to_datetime(df['날짜']).dt.month
    df['주말'] = df['요일'].isin([5, 6]).astype(int)
    
    # 시즌 진행도 (0~1)
    df['시즌진행도'] = df.groupby('연도')['날짜'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    
    return df

def create_streak_features(df, target_date=None):
    """연승/연패 관련 특성 생성 함수"""
    df = df.copy()
    df['연승연패'] = 0
    
    for team in df['팀명'].unique():
        team_mask = df['팀명'] == team
        team_data = df[team_mask].sort_values('날짜')
        
        # 현재 연승/연패 계산
        streak = 0
        streaks = [0]  # 첫 경기는 0으로 초기화
        
        # 이전 경기 결과로 연승/연패 계산
        prev_results = team_data['승리여부'].iloc[:-1]
        for result in prev_results:
            if result == '승':
                streak = max(1, streak + 1)
            elif result == '패':
                streak = min(-1, streak - 1)
            else:
                streak = 0
            streaks.append(streak)
        
        df.loc[team_mask, '연승연패'] = streaks
    
    return df

def create_head_to_head_features(df, target_date=None):
    """상대 전적 관련 특성 생성 함수"""
    df = df.copy()
    df['경기ID'] = df.groupby(['날짜', '경기장']).ngroup()
    
    # 상대팀 정보 추가
    df['상대팀'] = df.groupby('경기ID')['팀명'].transform(
        lambda x: x.iloc[::-1].values if len(x) == 2 else None
    )
    
    # 모든 팀 조합에 대한 승률 계산
    for team in df['팀명'].unique():
        for opponent in df['팀명'].unique():
            if team != opponent:
                col_name = f'VS_{opponent}_승률'
                df[col_name] = 0.5  # 기본값
                
                team_vs_opp = df[
                    (df['팀명'] == team) & 
                    (df['상대팀'] == opponent)
                ]
                
                if len(team_vs_opp) > 0:
                    # 누적 승률 계산
                    wins = (team_vs_opp['승리여부'] == '승').cumsum()
                    games = range(1, len(team_vs_opp) + 1)
                    win_rates = wins / games
                    
                    # 승률 업데이트
                    df.loc[team_vs_opp.index, col_name] = win_rates
    
    return df

def create_recent_performance_features(df, target_date=None, windows=[3, 5, 10]):
    """최근 성적 관련 특성 생성 함수"""
    df = df.copy()
    
    # 팀별로 처리
    for team in df['팀명'].unique():
        team_mask = df['팀명'] == team
        team_data = df[team_mask].sort_values('날짜')
        
        for window in windows:
            # 승률
            df.loc[team_mask, f'최근{window}경기_승률'] = (
                team_data['승리여부'].eq('승')
                .rolling(window, min_periods=1)
                .mean()
                .fillna(0.5)
            )
            
            # 득점
            df.loc[team_mask, f'최근{window}경기_득점'] = (
                team_data['득점']
                .rolling(window, min_periods=1)
                .mean()
                .fillna(team_data['득점'].mean())
            )
            
            # 안타율
            hits = team_data['안타'].rolling(window, min_periods=1).sum()
            at_bats = team_data['타수'].rolling(window, min_periods=1).sum()
            df.loc[team_mask, f'최근{window}경기_안타율'] = (
                hits / at_bats
            ).fillna(team_data['안타'].sum() / team_data['타수'].sum())
            
            # 홈/원정 구분
            for location in ['홈', '원정']:
                loc_mask = team_data['홈/원정'] == location
                if loc_mask.any():
                    df.loc[team_mask & df['홈/원정'].eq(location), f'최근{window}경기_{location}_승률'] = (
                        team_data[loc_mask]['승리여부'].eq('승')
                        .rolling(window, min_periods=1)
                        .mean()
                        .fillna(0.5)
                    )
    
    return df

def create_momentum_features(df, target_date=None):
    """팀 모멘텀 관련 특성 생성 함수"""
    momentum_features = []
    
    for team in df['팀명'].unique():
        team_data = df[
            (df['팀명'] == team) &
            (df['날짜'] <= target_date if target_date else True)
        ].sort_values('날짜')
        
        # 이전 5경기 득점 추세 (기울기)
        team_data['최근득점추세'] = (
            team_data['득점'].shift(1)
            .rolling(5, min_periods=1)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            .fillna(0)
        )
        
        # 이전 5경기 안타 추세
        team_data['최근안타추세'] = (
            team_data['안타'].shift(1)
            .rolling(5, min_periods=1)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            .fillna(0)
        )
        
        momentum_features.append(team_data)
    
    return pd.concat(momentum_features)

def save_preprocessed_data(df, file_path='preprocessed_data.pkl'):
    """전처리된 데이터를 피클 파일로 저장하는 함수"""
    logging.info(f"전처리된 데이터를 {file_path}에 저장 중...")
    df.to_pickle(file_path)
    logging.info("데이터 저장 완료")

def load_preprocessed_data(file_path='preprocessed_data.pkl'):
    """저장된 전처리 데이터를 로드하는 함수"""
    try:
        logging.info(f"{file_path}에서 전처리된 데이터를 로드 중...")
        df = pd.read_pickle(file_path)
        logging.info(f"데이터 로드 완료. 크기: {df.shape}")
        return df
    except FileNotFoundError:
        logging.warning("저장된 전처리 데이터를 찾을 수 없습니다.")
        return None

def preprocess_data(df):
    """데이터 전처리를 수행하는 함수"""
    logging.info("데이터 전처리 시작...")
    
    try:
        # 날짜 변환
        df['날짜'] = pd.to_datetime(df['날짜'])
        df = df.sort_values('날짜')
        
        # 메모리 사용량 모니터링
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        logging.info(f"초기 메모리 사용량: {initial_memory:.2f} MB")
        
        # 시계열 특성 한 번에 생성
        df = create_time_features(df)
        df = create_streak_features(df)
        df = create_head_to_head_features(df)
        df = create_recent_performance_features(df)
        df = create_momentum_features(df)
        
        # 범주형 변수 인코딩
        le = LabelEncoder()
        categorical_columns = ['팀명', '경기장', '홈/원정']
        
        for col in categorical_columns:
            df[col + '_인코딩'] = le.fit_transform(df[col])
        
        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        logging.info(f"최종 메모리 사용량: {final_memory:.2f} MB")
        logging.info("데이터 전처리 완료")
        
        return df
        
    except Exception as e:
        logging.error(f"데이터 전처리 중 치명적 오류 발생: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def main():
    try:
        logging.info("모델 실행을 시작합니다.")
        
        # 1. 전처리된 데이터 확인
        df_processed = load_preprocessed_data()
        
        if df_processed is None:
            # 전처리된 데이터가 없는 경우 새로 처리
            df = load_data()
            logging.info(f"데이터 로드 완료. 크기: {df.shape}")
            
            df_processed = preprocess_data(df)
            
            # 전처리된 데이터 저장
            save_preprocessed_data(df_processed)
        
        logging.info(f"전처리된 데이터 크기: {df_processed.shape}")
        
        # 3. 특성 선택
        features = [col for col in df_processed.columns if any(x in col for x in [
            '인코딩', '최근', '연승연패', 'VS_', '시즌진행도', '추세'
        ])]
        logging.info(f"선택된 특성 수: {len(features)}")
        logging.info(f"사용할 특성들: {', '.join(features)}")
        
        # 4. 데이터 분할
        X = df_processed[features]
        y = (df_processed['승리여부'] == '승').astype(int)
        
        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(X))
        train_index, test_index = splits[-1]
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        logging.info(f"학습 데이터 크기: {X_train.shape}")
        logging.info(f"테스트 데이터 크기: {X_test.shape}")
        
        # 5. 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 6. 모델 학습
        logging.info("모델 학습 시작...")
        model = xgb.XGBClassifier(
            learning_rate=0.02,
            max_depth=4,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # 기본 설정으로 학습
        model.fit(X_train_scaled, y_train)
        logging.info("모델 학습 완료")
        
        # 7. 예측
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # 8. 성능 평가
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        logging.info(f'시계열 특화 모델 정확도: {accuracy:.4f}')
        logging.info(f'시계열 특화 모델 AUC: {auc:.4f}')
        
        # 9. 특성 중요도 시각화
        plt.figure(figsize=(12, 6))
        importance = pd.Series(model.feature_importances_, index=features)
        importance.sort_values().plot(kind='barh')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance_temporal.png')
        plt.close()
        
        # 10. 결과 저장
        results = pd.DataFrame({
            '실제값': y_test,
            '예측값': y_pred,
            '예측확률': y_pred_proba
        })
        results.to_csv('prediction_results_temporal.csv', index=False, encoding='utf-8-sig')
        
        logging.info("모든 과정이 성공적으로 완료되었습니다.")
        
    except Exception as e:
        logging.error(f"오류 발생: {str(e)}")
        logging.error("상세 오류 정보:")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 