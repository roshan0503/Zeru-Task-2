import streamlit as st
import pandas as pd
import requests
import time
import numpy as np
from datetime import datetime
import io
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# App configuration
st.set_page_config(
    page_title="Compound Protocol Wallet Risk Scorer",
    page_icon="🏦",
    layout="wide"
)

# Constants
COVALENT_BASE_URL = "https://api.covalenthq.com/v1"
COMPOUND_V2_CONTRACTS = {
    'cETH': '0x4ddc2d193948926d02f9b1fe9e1daa0718270ed5',
    'cDAI': '0x5d3a536e4d6dbd6114cc1ead35777bab948e3643',
    'cUSDC': '0x39aa39c021dfbae8fac545936693ac917d5e7563',
    'cUSDT': '0xf650c3d88d12db855b8bf7d11be6c55a4e07dcc9',
    'cWBTC': '0xc11b1268c1a384e55c48c2391d8d480264a3a7f4'
}

class CompoundRiskScorer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def fetch_wallet_transactions(self, wallet_address: str) -> List[Dict]:
        """Fetch all transactions for a wallet address using Covalent API"""
        try:
            # Use Covalent's transaction endpoint
            url = f"{COVALENT_BASE_URL}/1/address/{wallet_address}/transactions_v2/"
            params = {
                'page-size': 1000,  # Maximum allowed
                'no-logs': False
            }
            
            if self.api_key:
                params['key'] = self.api_key
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    return data['data']['items']
            elif response.status_code == 429:
                st.warning("Rate limit reached. Waiting...")
                time.sleep(2)
                return []
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
                return []
                
        except Exception as e:
            st.error(f"Error fetching data for {wallet_address}: {str(e)}")
            return []
        
        return []
    
    def parse_compound_transactions(self, transactions: List[Dict], wallet_address: str) -> List[Dict]:
        """Parse transactions to identify Compound protocol interactions"""
        compound_txs = []
        
        for tx in transactions:
            if not tx.get('successful', False):
                continue
                
            tx_hash = tx.get('tx_hash', '')
            timestamp = tx.get('block_signed_at', '')
            gas_used = tx.get('gas_spent', 0)
            
            # Check if transaction interacts with known Compound contracts
            to_address = tx.get('to_address', '').lower()
            
            compound_interaction = None
            token_symbol = None
            
            # Check against known Compound V2 contracts
            for symbol, contract_addr in COMPOUND_V2_CONTRACTS.items():
                if to_address == contract_addr.lower():
                    compound_interaction = 'compound_v2'
                    token_symbol = symbol
                    break
            
            # Parse log events to determine transaction type
            tx_type = 'unknown'
            amount = 0
            
            if tx.get('log_events'):
                for log in tx['log_events']:
                    if log.get('decoded'):
                        event_name = log['decoded'].get('name', '').lower()
                        if 'mint' in event_name:
                            tx_type = 'supply'
                        elif 'redeem' in event_name:
                            tx_type = 'withdraw'
                        elif 'borrow' in event_name:
                            tx_type = 'borrow'
                        elif 'repay' in event_name:
                            tx_type = 'repay'
                        elif 'liquidate' in event_name:
                            tx_type = 'liquidation'
                        
                        # Try to extract amount
                        if log['decoded'].get('params'):
                            for param in log['decoded']['params']:
                                if param.get('name') in ['amount', 'mintAmount', 'redeemAmount']:
                                    amount = float(param.get('value', 0))
            
            if compound_interaction and tx_type != 'unknown':
                compound_txs.append({
                    'wallet_address': wallet_address,
                    'tx_hash': tx_hash,
                    'timestamp': timestamp,
                    'tx_type': tx_type,
                    'token': token_symbol or 'unknown',
                    'amount': amount,
                    'gas_used': gas_used,
                    'protocol_version': compound_interaction
                })
        
        return compound_txs
    
    def calculate_wallet_features(self, transactions: List[Dict]) -> Dict:
        """Calculate risk features for a wallet based on its Compound transactions"""
        if not transactions:
            return self._get_default_features()
        
        df = pd.DataFrame(transactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        df['gas_used'] = pd.to_numeric(df['gas_used'], errors='coerce').fillna(0)
        
        # Basic transaction metrics
        total_transactions = len(df)
        unique_tokens = df['token'].nunique()
        
        # Supply vs Borrow analysis
        supply_txs = df[df['tx_type'].isin(['supply', 'mint'])]
        borrow_txs = df[df['tx_type'] == 'borrow']
        repay_txs = df[df['tx_type'] == 'repay']
        liquidation_txs = df[df['tx_type'] == 'liquidation']
        
        total_supplied = supply_txs['amount'].sum()
        total_borrowed = borrow_txs['amount'].sum()
        total_repaid = repay_txs['amount'].sum()
        
        net_position = total_supplied - (total_borrowed - total_repaid)
        borrow_to_supply_ratio = total_borrowed / max(total_supplied, 1)
        
        # Time-based features
        first_tx_date = df['timestamp'].min()
        last_tx_date = df['timestamp'].max()
        account_age_days = (datetime.now() - first_tx_date).days if pd.notna(first_tx_date) else 0
        
        # Gas usage
        avg_gas_used = df['gas_used'].mean()
        
        # Liquidation risk indicators
        num_liquidations = len(liquidation_txs)
        liquidation_frequency = num_liquidations / max(total_transactions, 1)
        
        return {
            'total_transactions': total_transactions,
            'total_supplied': total_supplied,
            'total_borrowed': total_borrowed,
            'net_position': net_position,
            'num_liquidations': num_liquidations,
            'liquidation_frequency': liquidation_frequency,
            'account_age_days': account_age_days,
            'unique_tokens': unique_tokens,
            'borrow_to_supply_ratio': borrow_to_supply_ratio,
            'avg_gas_used': avg_gas_used
        }
    
    def _get_default_features(self) -> Dict:
        """Return default features for wallets with no Compound activity"""
        return {
            'total_transactions': 0,
            'total_supplied': 0,
            'total_borrowed': 0,
            'net_position': 0,
            'num_liquidations': 0,
            'liquidation_frequency': 0,
            'account_age_days': 0,
            'unique_tokens': 0,
            'borrow_to_supply_ratio': 0,
            'avg_gas_used': 0
        }
    
    def normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using Min-Max scaling"""
        normalized_df = features_df.copy()
        
        # Features to normalize (excluding wallet_address)
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            min_val = features_df[col].min()
            max_val = features_df[col].max()
            
            if max_val > min_val:
                normalized_df[f'{col}_norm'] = (features_df[col] - min_val) / (max_val - min_val)
            else:
                normalized_df[f'{col}_norm'] = 0
        
        return normalized_df
    
    def calculate_risk_score(self, features: Dict) -> Tuple[float, Dict]:
        """Calculate risk score based on normalized features"""
        
        # Risk scoring weights (higher weight = more influence on risk)
        weights = {
            'liquidation_frequency': 300,      # High liquidation frequency = high risk
            'borrow_to_supply_ratio': 250,     # High leverage = high risk
            'num_liquidations': 200,           # More liquidations = higher risk
            'total_transactions': -50,         # More activity = lower risk (negative weight)
            'account_age_days': -100,          # Older accounts = lower risk
            'unique_tokens': -30,              # Diversification = lower risk
            'net_position': -80,               # Positive net position = lower risk
            'avg_gas_used': 20                 # High gas usage might indicate risky behavior
        }
        
        # Normalize individual features for scoring
        score_components = {}
        
        # Liquidation frequency (0-1 range)
        liq_freq = min(features['liquidation_frequency'], 1.0)
        score_components['liquidation_risk'] = liq_freq * weights['liquidation_frequency']
        
        # Borrow to supply ratio (cap at 2.0 for scoring)
        borrow_ratio = min(features['borrow_to_supply_ratio'], 2.0) / 2.0
        score_components['leverage_risk'] = borrow_ratio * weights['borrow_to_supply_ratio']
        
        # Number of liquidations (normalize by log scale)
        num_liq_norm = min(np.log1p(features['num_liquidations']) / np.log1p(10), 1.0)
        score_components['liquidation_count_risk'] = num_liq_norm * weights['num_liquidations']
        
        # Activity level (more transactions = lower risk)
        activity_norm = min(np.log1p(features['total_transactions']) / np.log1p(100), 1.0)
        score_components['activity_benefit'] = activity_norm * weights['total_transactions']
        
        # Account age (older = lower risk)
        age_norm = min(features['account_age_days'] / 365, 3.0) / 3.0
        score_components['age_benefit'] = age_norm * weights['account_age_days']
        
        # Token diversification
        token_norm = min(features['unique_tokens'] / 10, 1.0)
        score_components['diversification_benefit'] = token_norm * weights['unique_tokens']
        
        # Net position (positive = lower risk)
        if features['total_supplied'] > 0:
            net_pos_norm = max(min(features['net_position'] / features['total_supplied'], 1.0), -1.0)
        else:
            net_pos_norm = 0
        score_components['net_position_benefit'] = net_pos_norm * weights['net_position']
        
        # Gas usage risk
        gas_norm = min(features['avg_gas_used'] / 1000000, 1.0) if features['avg_gas_used'] > 0 else 0
        score_components['gas_risk'] = gas_norm * weights['avg_gas_used']
        
        # Calculate final score (0-1000 scale)
        raw_score = sum(score_components.values())
        
        # Ensure score is between 0 and 1000
        risk_score = max(0, min(1000, 500 + raw_score))
        
        return risk_score, score_components

def main():
    st.title("🏦 Compound Protocol Wallet Risk Scorer")
    st.markdown("""
    This application analyzes wallet interactions with Compound V2/V3 protocols and assigns risk scores from 0-1000.
    
    **Risk Score Interpretation:**
    - **0-300**: Low Risk (Conservative DeFi users)
    - **301-600**: Medium Risk (Active traders)
    - **601-1000**: High Risk (High leverage, frequent liquidations)
    """)
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input(
            "Covalent API Key (Optional)", 
            type="password",
            help="Enter your free Covalent API key for higher rate limits. Get one at covalenthq.com"
        )
        
        st.markdown("### 📊 Risk Factors")
        st.markdown("""
        **High Risk Indicators:**
        - Frequent liquidations
        - High borrow-to-supply ratio
        - Recent account creation
        - Single token exposure
        
        **Low Risk Indicators:**
        - Long account history
        - Diversified token portfolio
        - Positive net position
        - No liquidation history
        """)
    
    # File upload section
    st.header("📁 Upload Wallet Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file with wallet addresses",
        type=['csv'],
        help="CSV should contain a 'wallet_id' column with Ethereum addresses"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Validate required column
            if 'wallet_id' not in df.columns:
                st.error("❌ CSV must contain a 'wallet_id' column with Ethereum addresses")
                return
            
            # Display preview
            st.success(f"✅ Loaded {len(df)} wallet addresses")
            st.dataframe(df.head(), use_container_width=True)
            
            # Process wallets button
            if st.button("🚀 Analyze Wallets", type="primary"):
                scorer = CompoundRiskScorer(api_key)
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                total_wallets = len(df)
                
                for idx, wallet_address in enumerate(df['wallet_id']):
                    status_text.text(f"Processing wallet {idx+1}/{total_wallets}: {wallet_address}")
                    
                    # Fetch and analyze wallet
                    transactions = scorer.fetch_wallet_transactions(wallet_address)
                    compound_txs = scorer.parse_compound_transactions(transactions, wallet_address)
                    features = scorer.calculate_wallet_features(compound_txs)
                    risk_score, score_components = scorer.calculate_risk_score(features)
                    
                    results.append({
                        'wallet_id': wallet_address,
                        'risk_score': round(risk_score, 2),
                        'total_transactions': features['total_transactions'],
                        'total_supplied': features['total_supplied'],
                        'total_borrowed': features['total_borrowed'],
                        'num_liquidations': features['num_liquidations'],
                        'account_age_days': features['account_age_days']
                    })
                    
                    progress_bar.progress((idx + 1) / total_wallets)
                    
                    # Rate limiting
                    if not api_key:
                        time.sleep(0.5)  # Respect free tier limits
                
                status_text.text("✅ Analysis Complete!")
                
                # Display results
                results_df = pd.DataFrame(results)
                
                st.header("📊 Risk Analysis Results")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average Risk Score", f"{results_df['risk_score'].mean():.1f}")
                with col2:
                    st.metric("High Risk Wallets", f"{len(results_df[results_df['risk_score'] > 600])}")
                with col3:
                    st.metric("Wallets with Activity", f"{len(results_df[results_df['total_transactions'] > 0])}")
                with col4:
                    st.metric("Wallets with Liquidations", f"{len(results_df[results_df['num_liquidations'] > 0])}")
                
                # Results table
                st.dataframe(
                    results_df.sort_values('risk_score', ascending=False),
                    use_container_width=True
                )
                
                # Download button
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name=f"compound_risk_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        # Show sample CSV format
        st.header("📋 Sample CSV Format")
        sample_df = pd.DataFrame({
            'wallet_id': [
                '0x742d35cc1ae0cd95a5e7b48f0a77b1b0ef0d5c7e',
                '0x8ba1f109551bd432803012645hac136c0c8f47eb',
                '0x40ec5b33f54e0e8a33a975908c5ba1c14e5bccdf'
            ]
        })
        st.dataframe(sample_df, use_container_width=True)

if __name__ == "__main__":
    main()
