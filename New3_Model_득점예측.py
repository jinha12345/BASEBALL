import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import logging
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # 데이터 로드
        logging.info("데이터 로드 시작...")
        try:
            df = pd.read_excel('BASEBALL_stats_15.xlsx')
            logging.info(f"데이터 로드 완료. 데이터 크기: {df.shape}")
            logging.info(f"컬럼 목록: {df.columns.tolist()}")
            
            # 필수 컬럼 확인
            required_columns = ['팀명', '경기장', '홈/원정', '날짜', '타수', '안타', '2루타', '3루타', '홈런', '볼넷', '사구', '득점']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
                
            logging.info("데이터 기본 정보:")
            logging.info(f"데이터 타입:\n{df.dtypes}")
            logging.info(f"결측치 개수:\n{df.isnull().sum()}")
            logging.info(f"팀명 고유값:\n{df['팀명'].unique()}")
            logging.info(f"경기장 고유값:\n{df['경기장'].unique()}")
            
        except Exception as e:
            logging.error(f"데이터 로드 중 오류 발생: {str(e)}")
            raise

        # 데이터 전처리
        logging.info("데이터 전처리 시작...")
        
        # 날짜 형식 변환
        logging.info("날짜 형식 변환 중...")
        df['날짜'] = pd.to_datetime(df['날짜'])
        
        # 경기 ID 생성 (날짜와 경기장 기준으로)
        logging.info("경기 ID 생성 중...")
        df['경기ID'] = df.groupby(['날짜', '경기장']).ngroup()
        
        # 상대팀 정보 추가
        logging.info("상대팀 정보 추가 중...")
        df['상대팀'] = df.groupby(['경기ID'])['팀명'].transform(lambda x: x.iloc[::-1].values if len(x) == 2 else None)
        
        # 팀명 정규화
        logging.info("팀명 정규화 중...")
        def normalize_team(team):
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
            for old_name, new_name in team_mapping.items():
                if old_name in team:
                    return new_name
            return '기타'
        
        df['팀명'] = df['팀명'].apply(normalize_team)
        df['상대팀'] = df['상대팀'].apply(normalize_team)
        logging.info(f"고유 팀명 목록: {df['팀명'].unique().tolist()}")
        logging.info(f"고유 상대팀 목록: {df['상대팀'].unique().tolist()}")
        
        # 경기장 이름 정규화
        logging.info("경기장 정규화 중...")
        def normalize_stadium(stadium):
            stadium = str(stadium).strip()
            if '잠실' in stadium:
                return '잠실'
            elif '고척' in stadium:
                return '고척'
            elif '문학' in stadium:
                return '문학'
            elif '수원' in stadium:
                return '수원'
            elif '대구' in stadium:
                return '대구'
            elif '사직' in stadium:
                return '사직'
            elif '광주' in stadium:
                return '광주'
            elif '대전' in stadium:
                return '대전'
            elif '창원' in stadium or '마산' in stadium:
                return '창원'
            else:
                return '기타'
        
        df['경기장'] = df['경기장'].apply(normalize_stadium)
        logging.info(f"고유 경기장 목록: {df['경기장'].unique().tolist()}")
        
        # 홈/원정 정규화
        logging.info("홈/원정 정규화 중...")
        def normalize_home_away(value):
            value = str(value).strip()
            if '홈' in value:
                return '홈'
            elif '원정' in value:
                return '원정'
            else:
                return '알수없음'
                
        df['홈/원정'] = df['홈/원정'].apply(normalize_home_away)
        logging.info(f"홈/원정 구분: {df['홈/원정'].unique().tolist()}")
        
        # 장타율 계산을 위한 루타 계산
        logging.info("\n루타 계산 중...")
        df['루타'] = df['안타'] + df['2루타']*2 + df['3루타']*3 + df['홈런']*4
        
        # 승리여부 계산
        logging.info("승리여부 계산 중...")
        def determine_result(group):
            if len(group) != 2:
                return ['무'] * len(group)
            score1, score2 = group['득점'].values
            if score1 > score2:
                return ['승', '패']
            elif score1 < score2:
                return ['패', '승']
            return ['무', '무']

        logging.info("승리여부 계산 중...")
        df['승리여부'] = df.groupby('경기ID').apply(
            lambda x: pd.Series(determine_result(x), index=x.index)
        ).values

        def calculate_team_stats(df, window=10):
            logging.info(f"팀 통계 계산 중 (window={window})...")
            stats_list = []
            
            for team in df['팀명'].unique():
                team_data = df[df['팀명'] == team].sort_values('날짜')
                
                # 통계 계산
                stats = pd.DataFrame({
                    '최근승률': team_data['승리여부'].eq('승').rolling(window, min_periods=1).mean(),
                    '최근평균득점': team_data['득점'].rolling(window, min_periods=1).mean(),
                    '최근평균타율': team_data['안타'].rolling(window, min_periods=1).sum() / 
                                team_data['타수'].rolling(window, min_periods=1).sum(),
                    '최근평균출루율': (team_data['안타'] + team_data['볼넷'] + team_data['사구']).rolling(window, min_periods=1).sum() / 
                                (team_data['타수'] + team_data['볼넷'] + team_data['사구']).rolling(window, min_periods=1).sum(),
                    '최근평균장타율': team_data['루타'].rolling(window, min_periods=1).sum() / 
                                team_data['타수'].rolling(window, min_periods=1).sum()
                })
                
                stats['팀명'] = team
                stats['날짜'] = team_data['날짜']
                stats_list.append(stats)
            
            return pd.concat(stats_list)

        # 팀 통계 계산
        team_stats = calculate_team_stats(df)
        df = df.merge(team_stats, on=['팀명', '날짜'], how='left')

        # 상대팀 통계 계산
        logging.info("상대팀 통계 계산 중...")
        opponent_stats = team_stats.copy()
        opponent_stats.columns = ['상대팀' + col if col not in ['팀명', '날짜'] else col 
                                for col in opponent_stats.columns]
        opponent_stats = opponent_stats.rename(columns={'팀명': '상대팀'})
        df = df.merge(opponent_stats, on=['상대팀', '날짜'], how='left')

        # 시간 관련 피처 생성
        logging.info("시간 관련 피처 생성 중...")
        df['주말'] = df['날짜'].dt.dayofweek.isin([5, 6]).astype(int)
        df['월'] = df['날짜'].dt.month

        # 범주형 변수 인코딩
        logging.info("범주형 변수 인코딩 중...")
        le = LabelEncoder()
        categorical_columns = ['팀명', '상대팀', '경기장', '홈/원정']
        for col in categorical_columns:
            # 데이터 전처리: 문자열로 변환하고 공백 제거
            df[col] = df[col].astype(str).str.strip()
            # NaN 값을 '알수없음'으로 대체
            df[col] = df[col].replace('nan', '알수없음')
            df[col + '_인코딩'] = le.fit_transform(df[col])

        # 최종 특성 선택 (득점 예측용)
        features = ['팀명_인코딩', '상대팀_인코딩', '경기장_인코딩', '홈/원정_인코딩',
                   '최근승률', '최근평균득점', '최근평균타율', '최근평균출루율', '최근평균장타율',
                   '상대팀최근승률', '상대팀최근평균득점', '주말', '월']

        # 목표 변수를 승리여부에서 득점으로 변경
        original_wins = (df['승리여부'] == '승').astype(int)  # 원래 승패 정보 저장
        y = df['득점']  # 새로운 목표 변수: 득점

        # 데이터 분할
        logging.info("\n데이터 분할 중...")
        X = df[features]

        # 시계열 분할
        tscv = TimeSeriesSplit(n_splits=5)
        splits = list(tscv.split(X))
        train_index, test_index = splits[-1]  # 마지막 분할 사용

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        original_wins_test = original_wins.iloc[test_index]  # 테스트 세트의 실제 승패

        logging.info(f"학습 데이터 크기: {X_train.shape}")
        logging.info(f"테스트 데이터 크기: {X_test.shape}")

        # 스케일링
        logging.info("\n스케일링 중...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 모델 학습 (회귀 모델로 변경)
        logging.info("\n모델 학습 중...")
        
        # XGBoost 회귀 모델
        logging.info("XGBoost 모델 학습...")
        xgb_model = xgb.XGBRegressor(
            learning_rate=0.03,
            max_depth=5,
            n_estimators=300,
            objective='reg:squarederror',
            random_state=42
        )
        xgb_model.fit(X_train_scaled, y_train)

        # LightGBM 회귀 모델
        logging.info("LightGBM 모델 학습...")
        lgb_model = lgb.LGBMRegressor(
            learning_rate=0.03,
            max_depth=5,
            n_estimators=300,
            objective='regression',
            random_state=42
        )
        lgb_model.fit(X_train_scaled, y_train)

        # 득점 예측
        logging.info("\n예측 수행 중...")
        xgb_pred = xgb_model.predict(X_test_scaled)
        lgb_pred = lgb_model.predict(X_test_scaled)

        # 앙상블 득점 예측
        ensemble_pred = 0.6 * xgb_pred + 0.4 * lgb_pred

        # 득점 예측 성능 평가
        mse = mean_squared_error(y_test, ensemble_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, ensemble_pred)
        
        logging.info(f'\n득점 예측 RMSE: {rmse:.4f}')
        logging.info(f'득점 예측 R2 Score: {r2:.4f}')

        # 승패 예측을 위해 같은 경기의 양 팀 득점 예측값 비교
        logging.info("\n승패 예측 변환 중...")
        
        # 경기별로 예측된 득점을 그룹화하여 승패 결정
        df_test = df.iloc[test_index].copy()
        df_test['예측득점'] = ensemble_pred
        
        def determine_win_from_predicted_scores(group):
            if len(group) != 2:
                return [0] * len(group)
            score1, score2 = group['예측득점'].values
            if score1 > score2:
                return [1, 0]
            elif score1 < score2:
                return [0, 1]
            return [0, 0]  # 동점인 경우

        predicted_wins = df_test.groupby('경기ID').apply(
            lambda x: pd.Series(determine_win_from_predicted_scores(x), index=x.index)
        ).values

        # 승패 예측 성능 평가
        accuracy = accuracy_score(original_wins_test, predicted_wins)
        auc = roc_auc_score(original_wins_test, predicted_wins)

        logging.info(f'\n승패 예측 정확도: {accuracy:.4f}')
        logging.info(f'승패 예측 AUC: {auc:.4f}')

        # 특성 중요도 시각화
        logging.info("\n특성 중요도 시각화 중...")
        
        # 특성 이름을 한글로 변경
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
        xgb_importance = pd.Series(xgb_model.feature_importances_, 
                                 index=[feature_names_korean[f] for f in features])
        xgb_importance.sort_values(ascending=True).plot(kind='barh')
        plt.title('XGBoost 득점 예측 특성 중요도', fontsize=14, pad=20)
        plt.xlabel('중요도', fontsize=12)
        plt.ylabel('특성', fontsize=12)
        plt.tight_layout()
        plt.savefig('feature_importance_scoring.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 결과 저장
        logging.info("\n결과 저장 중...")
        results = pd.DataFrame({
            '실제득점': y_test,
            '예측득점': ensemble_pred,
            '실제승패': original_wins_test,
            '예측승패': predicted_wins
        })
        results.to_csv('prediction_results_scoring.csv', index=False, encoding='utf-8-sig')
        
        logging.info("\n모든 과정이 완료되었습니다.")
        
    except Exception as e:
        logging.error(f"\n오류 발생: {str(e)}")
        import traceback
        logging.error("\n상세 오류 정보:")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 