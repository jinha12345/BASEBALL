import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import logging
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
import traceback

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # MacOS 용 폰트로 변경
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path='BASEBALL_stats_15.xlsx'):
    """
    데이터를 로드하고 기본 정보를 출력하는 함수
    """
    logging.info("데이터 로드 시작...")
    try:
        df = pd.read_excel(file_path)
        logging.info(f"데이터 로드 완료. 데이터 크기: {df.shape}")
        
        # 필수 컬럼 확인
        required_columns = ['팀명', '경기장', '홈/원정', '날짜', '타수', '안타', '득점']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
            
        # 데이터 기본 정보 출력
        logging.info(f"컬럼 목록: {df.columns.tolist()}")
        logging.info("데이터 기본 정보:")
        logging.info(f"데이터 타입:\n{df.dtypes}")
        logging.info(f"결측치 개수:\n{df.isnull().sum()}")
        
        return df
        
    except Exception as e:
        logging.error(f"데이터 로드 중 오류 발생: {str(e)}")
        raise

def normalize_team_names(team):
    """
    팀명을 정규화하는 함수
    """
    team = str(team).strip()
    team_mapping = {
        '넥센': '히어로즈',
        '키움': '히어로즈',
        'SK': 'SSG',
        'KT': 'KT',
        'NC': 'NC',
        'LG': 'LG',
        '두산': '두산',
        '롯데': '롯데',
        '삼성': '삼성',
        '한화': '한화',
        'KIA': 'KIA'
    }
    return team_mapping.get(team, '기타')

def normalize_stadium(stadium):
    """
    경기장 이름을 정규화하는 함수
    """
    stadium = str(stadium).strip()
    stadium_mapping = {
        '잠실': ['잠실'],
        '고척': ['고척'],
        '문학': ['문학'],
        '수원': ['수원'],
        '대구': ['대구'],
        '사직': ['사직'],
        '광주': ['광주'],
        '대전': ['대전'],
        '창원': ['창원', '마산']
    }
    
    for normalized, variants in stadium_mapping.items():
        if any(variant in stadium for variant in variants):
            return normalized
    return '기타'

def normalize_home_away(value):
    """
    홈/원정 구분을 정규화하는 함수
    """
    value = str(value).strip()
    if '홈' in value:
        return '홈'
    elif '원정' in value:
        return '원정'
    return '알수없음'

def calculate_team_stats(df, target_date, window=10):
    """
    특정 날짜 이전의 데이터만 사용하여 팀별 통계를 계산하는 함수
    """
    logging.info(f"팀 통계 계산 중 (기준일: {target_date}, window={window})...")
    stats_list = []
    
    for team in df['팀명'].unique():
        # 해당 날짜 이전의 데이터만 선택
        team_data = df[
            (df['팀명'] == team) & 
            (df['날짜'] < target_date)
        ].sort_values('날짜')
        
        # 데이터가 충분하지 않은 경우 처리
        if len(team_data) < 1:
            continue
            
        # 통계 계산
        recent_stats = pd.DataFrame({
            '최근승률': team_data['승리여부'].eq('승').rolling(window, min_periods=1).mean().iloc[-1],
            '최근평균득점': team_data['득점'].rolling(window, min_periods=1).mean().iloc[-1],
            '최근평균타율': (
                team_data['안타'].rolling(window, min_periods=1).sum() /
                team_data['타수'].rolling(window, min_periods=1).sum()
            ).iloc[-1],
            '최근평균출루율': (
                (team_data['안타'] + team_data['볼넷'] + team_data['사구']).rolling(window, min_periods=1).sum() /
                (team_data['타수'] + team_data['볼넷'] + team_data['사구']).rolling(window, min_periods=1).sum()
            ).iloc[-1],
            '최근평균장타율': (
                team_data['루타'].rolling(window, min_periods=1).sum() /
                team_data['타수'].rolling(window, min_periods=1).sum()
            ).iloc[-1]
        }, index=[0])
        
        recent_stats['팀명'] = team
        recent_stats['날짜'] = target_date
        stats_list.append(recent_stats)
    
    if not stats_list:
        return pd.DataFrame()
    
    return pd.concat(stats_list, ignore_index=True)

def determine_match_result(group):
    """
    경기별 승패를 결정하는 함수
    """
    if len(group) != 2:
        return ['무'] * len(group)
    score1, score2 = group['득점'].values
    if score1 > score2:
        return ['승', '패']
    elif score1 < score2:
        return ['패', '승']
    return ['무', '무']

def preprocess_data(df):
    """
    데이터 전처리를 수행하는 함수
    """
    logging.info("데이터 전처리 시작...")
    
    # 기본 전처리 (날짜 변환, 정규화 등)
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.sort_values('날짜')
    
    # 경기 ID 및 기본 정보 처리
    df['경기ID'] = df.groupby(['날짜', '경기장']).ngroup()
    df['상대팀'] = df.groupby(['경기ID'])['팀명'].transform(
        lambda x: x.iloc[::-1].values if len(x) == 2 else None)
    
    # 정규화
    df['팀명'] = df['팀명'].apply(normalize_team_names)
    df['상대팀'] = df['상대팀'].apply(normalize_team_names)
    df['경기장'] = df['경기장'].apply(normalize_stadium)
    df['홈/원정'] = df['홈/원정'].apply(normalize_home_away)
    
    # 루타 계산
    df['루타'] = df['안타'] + df['2루타']*2 + df['3루타']*3 + df['홈런']*4
    
    # 승리여부 계산
    df['승리여부'] = df.groupby('경기ID').apply(
        lambda x: pd.Series(determine_match_result(x), index=x.index)
    ).values
    
    # 각 날짜별로 통계 계산
    processed_data = []
    for date in df['날짜'].unique():
        # 해당 날짜의 경기 데이터
        current_games = df[df['날짜'] == date]
        
        # 해당 날짜 이전 데이터로 통계 계산
        team_stats = calculate_team_stats(df, date)
        
        # 현재 경기 데이터와 통계 병합
        if not team_stats.empty:
            current_games = current_games.merge(
                team_stats, on=['팀명', '날짜'], how='left')
            
            # 상대팀 통계 병합
            opponent_stats = team_stats.copy()
            opponent_stats.columns = [
                '상대팀' + col if col not in ['팀명', '날짜'] else col 
                for col in opponent_stats.columns
            ]
            opponent_stats = opponent_stats.rename(columns={'팀명': '상대팀'})
            current_games = current_games.merge(
                opponent_stats, on=['상대팀', '날짜'], how='left')
        
        processed_data.append(current_games)
    
    # 모든 처리된 데이터 합치기
    df_processed = pd.concat(processed_data, ignore_index=True)
    
    # 시간 관련 피처 생성
    df_processed['주말'] = df_processed['날짜'].dt.dayofweek.isin([5, 6]).astype(int)
    df_processed['월'] = df_processed['날짜'].dt.month
    
    return df_processed

def encode_categorical_features(df):
    """
    범주형 변수를 인코딩하는 함수
    """
    logging.info("범주형 변수 인코딩 중...")
    le = LabelEncoder()
    categorical_columns = ['팀명', '상대팀', '경기장', '홈/원정']
    
    for col in categorical_columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace('nan', '알수없음')
        df[col + '_인코딩'] = le.fit_transform(df[col])
    
    return df

def train_models(X_train_scaled, y_train):
    """
    XGBoost와 LightGBM 모델을 학습하는 함수
    """
    logging.info("\n모델 학습 중...")
    
    # XGBoost 모델
    logging.info("XGBoost 모델 학습...")
    xgb_model = xgb.XGBClassifier(
        learning_rate=0.03,
        max_depth=5,
        n_estimators=300,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    # LightGBM 모델
    logging.info("LightGBM 모델 학습...")
    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.01,
        'max_depth': 4,
        'num_leaves': 15,
        'min_child_samples': 20,
        'min_child_weight': 1e-3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_gain_to_split': 1e-7,
        'min_data_in_leaf': 20,
        'max_bin': 255,
        'verbose': -1,
        'random_state': 42,
        'force_col_wise': True
    }
    
    lgb_model = lgb.train(
        params,
        train_data,
        num_boost_round=200
    )
    
    return xgb_model, lgb_model

def visualize_feature_importance(model, features, output_file):
    """
    특성 중요도를 시각화하는 함수
    """
    feature_names_korean = {
        '팀명_인코딩': '팀명',
        '상대팀_인코딩': '상대팀',
        '경기장_인코딩': '경기장',
        '홈/원정_인코딩': '홈/원정',
        '최근승률': '최근 승률',
        '최근평균득점': '최근 평균 득점',
        '최근평균타율': '최근 평균 타율',
        '최근평균출루율': '최근 평균 출루율',
        '최근평균장타율': '최근 평균 장타율',
        '상대팀최근승률': '상대팀 최근 승률',
        '상대팀최근평균득점': '상대팀 최근 평균 득점',
        '주말': '주말 경기',
        '월': '월별 경기'
    }
    
    plt.figure(figsize=(12, 8))
    importance = pd.Series(model.feature_importances_, 
                         index=[feature_names_korean.get(f, f) for f in features])
    importance.sort_values(ascending=True).plot(kind='barh')
    plt.title('XGBoost 특성 중요도', fontsize=14, pad=20)
    plt.xlabel('중요도', fontsize=12)
    plt.ylabel('특성', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        # 1. 데이터 로드
        df = load_data()
        
        # 2. 데이터 전처리
        df = preprocess_data(df)
        
        # 3. 범주형 변수 인코딩
        df = encode_categorical_features(df)
        
        # 4. 특성 선택
        features = ['팀명_인코딩', '상대팀_인코딩', '경기장_인코딩', '홈/원정_인코딩',
                   '최근승률', '최근평균득점', '최근평균타율', '최근평균출루율', '최근평균장타율',
                   '상대팀최근승률', '상대팀최근평균득점', '주말', '월']
        
        # 5. 데이터 분할 (시계열 고려)
        X = df[features]
        y = (df['승리여부'] == '승').astype(int)
        
        # 시계열 분할
        train_size = int(len(df) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # 6. 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 7. 모델 학습
        xgb_model, lgb_model = train_models(X_train_scaled, y_train)
        
        # 8. 예측
        xgb_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
        lgb_pred_proba = lgb_model.predict(X_test_scaled)
        
        # 9. 앙상블
        ensemble_pred_proba = 0.6 * xgb_pred_proba + 0.4 * lgb_pred_proba
        ensemble_predictions = (ensemble_pred_proba > 0.5).astype(int)
        
        # 10. 성능 평가
        accuracy = accuracy_score(y_test, ensemble_predictions)
        auc = roc_auc_score(y_test, ensemble_pred_proba)
        
        logging.info(f'\n앙상블 모델 정확도: {accuracy:.4f}')
        logging.info(f'앙상블 모델 AUC: {auc:.4f}')
        
        # 11. 특성 중요도 시각화
        visualize_feature_importance(xgb_model, features, 'feature_importance.png')
        
        # 12. 결과 저장
        results = pd.DataFrame({
            '실제값': y_test,
            '예측값': ensemble_predictions,
            '예측확률': ensemble_pred_proba
        })
        results.to_csv('prediction_results.csv', index=False, encoding='utf-8-sig')
        
        logging.info("\n모든 과정이 완료되었습니다.")
        
    except Exception as e:
        logging.error(f"\n오류 발생: {str(e)}")
        logging.error("\n상세 오류 정보:")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()