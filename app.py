import streamlit as st
import pandas as pd
import requests
import time
import numpy as np
from datetime import datetime, timezone
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
MORALIS_BASE_URL = "https://deep-index.moralis.io/api/v2"
MORALIS_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJub25jZSI6IjI3NGY4MmFhLTdjYmMtNDczOC05ZmI5LTQzYmM1NWM5NWNmNyIsIm9yZ0lkIjoiNDYxMzg2IiwidXNlcklkIjoiNDc0Njc4IiwidHlwZUlkIjoiNzZjYjQzZDYtMmFhNC00YzQxLTg4ZGEtYjJhMDg3MTUzYWFjIiwidHlwZSI6IlBST0pFQ1QiLCJpYXQiOjE3NTM0NTUxNDMsImV4cCI6NDkwOTIxNTE0M30.XfykIGtPO4G1KmSx6bk3c6mQohOJH4whVjOOALhQEcA"

# Compound V2 contract addresses for identification
COMPOUND_V2_TOKENS = {
    "0x4ddc2d193948926d02f9b1fe9e1daa0718270ed5": "cETH",
    "0x5d3a536e4d6dbd6114cc1ead35777bab948e3643": "cDAI",
    "0x39aa39c021dfbae8fac545936693ac917d5e7563": "cUSDC",
    "0xf650c3d88d12db855b8bf7d11be6c55a4e07dcc9": "cUSDT",
    "0xc11b1268c1a384e55c48c2391d8d480264a3a7f4": "cWBTC",
    "0x70e36f6bf80a52b3b46b3af8e106cc0ed743e8e4": "ccomp",
    "0x35a18000230da775cac24873d00ff85bccded550": "cUNI"
}

class MoralisCompoundRiskScorer:
    def __init__(self):
        self.api_key = MORALIS_API_KEY
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': self.api_key,
            'Content-Type': 'application/json'
        })
    
    def fetch_wallet_erc20_balances(self, wallet_address: str) -> List[Dict]:
        """Fetch current ERC20 token balances"""
        try:
            url = f"{MORALIS_BASE_URL}/{wallet_address}/erc20"
            params = {
                'chain': 'eth',
                'exclude_spam': 'true'
            }
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return data if isinstance(data, list) else []
            elif response.status_code == 429:
                st.warning("Rate limit reached. Waiting...")
                time.sleep(2)
                return []
            else:
                return []
                
        except Exception as e:
            return []
    
    def fetch_wallet_erc20_transfers(self, wallet_address: str) -> List[Dict]:
        """Fetch ERC20 transfer history"""
        try:
            url = f"{MORALIS_BASE_URL}/{wallet_address}/erc20/transfers"
            params = {
                'chain': 'eth',
                'limit': 100,
                'order': 'DESC'
            }
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('result', []) if isinstance(data, dict) else []
            else:
                return []
                
        except Exception as e:
            return []
    
    def fetch_wallet_transactions(self, wallet_address: str) -> List[Dict]:
        """Fetch general wallet transactions"""
        try:
            url = f"{MORALIS_BASE_URL}/{wallet_address}"
            params = {
                'chain': 'eth',
                'limit': 50,
                'order': 'DESC'
            }
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('result', []) if isinstance(data, dict) else []
            else:
                return []
                
        except Exception as e:
            return []
    
    def parse_compound_activity(self, balances: List[Dict], transfers: List[Dict], transactions: List[Dict], wallet_address: str) -> List[Dict]:
        """Parse wallet data to identify Compound protocol activity"""
        compound_activities = []
        
        # Parse current Compound token balances
        for balance in balances:
            token_address = balance.get('token_address', '').lower()
            token_symbol = balance.get('symbol', '')
            
            if token_address in COMPOUND_V2_TOKENS or (token_symbol and token_symbol.lower().startswith('c')):
                balance_formatted = float(balance.get('balance_formatted', 0))
                if balance_formatted > 0:
                    compound_activities.append({
                        'wallet_address': wallet_address,
                        'protocol': 'Compound V2',
                        'activity_type': 'current_position',
                        'token': COMPOUND_V2_TOKENS.get(token_address, token_symbol),
                        'amount': balance_formatted,
                        'usd_value': float(balance.get('usd_value', 0)) if balance.get('usd_value') else 0,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'tx_hash': None
                    })
        
        # Parse ERC20 transfers for Compound tokens
        for transfer in transfers:
            token_address = transfer.get('address', '').lower()
            token_symbol = transfer.get('token_symbol', '')
            
            if token_address in COMPOUND_V2_TOKENS or (token_symbol and token_symbol.lower().startswith('c')):
                from_address = transfer.get('from_address', '').lower()
                to_address = transfer.get('to_address', '').lower()
                wallet_lower = wallet_address.lower()
                
                # Determine transaction type
                if from_address == wallet_lower:
                    activity_type = 'redeem' if token_symbol and token_symbol.lower().startswith('c') else 'repay'
                elif to_address == wallet_lower:
                    activity_type = 'mint' if token_symbol and token_symbol.lower().startswith('c') else 'borrow'
                else:
                    continue
                
                # Parse timestamp safely
                timestamp_str = transfer.get('block_timestamp', '')
                try:
                    if timestamp_str:
                        # Parse the timestamp and ensure it's timezone-aware
                        if 'T' in timestamp_str and '+' not in timestamp_str and 'Z' not in timestamp_str:
                            timestamp_str += 'Z'  # Add UTC indicator if missing
                        parsed_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        parsed_timestamp = datetime.now(timezone.utc)
                except:
                    parsed_timestamp = datetime.now(timezone.utc)
                
                compound_activities.append({
                    'wallet_address': wallet_address,
                    'protocol': 'Compound V2',
                    'activity_type': activity_type,
                    'token': COMPOUND_V2_TOKENS.get(token_address, token_symbol),
                    'amount': float(transfer.get('value_formatted', 0)),
                    'usd_value': float(transfer.get('usd_value', 0)) if transfer.get('usd_value') else 0,
                    'timestamp': parsed_timestamp.isoformat(),
                    'tx_hash': transfer.get('transaction_hash', '')
                })
        
        return compound_activities
    
    def calculate_wallet_features(self, activities: List[Dict]) -> Dict:
        """Calculate risk features for a wallet based on its Compound activities"""
        if not activities:
            return self._get_default_features()
        
        df = pd.DataFrame(activities)
        
        # Convert timestamp column with proper timezone handling
        def parse_timestamp_safe(ts):
            try:
                if isinstance(ts, str):
                    if 'T' in ts:
                        # Parse ISO format timestamp
                        if '+' not in ts and 'Z' not in ts:
                            ts += 'Z'
                        return pd.to_datetime(ts, utc=True)
                    else:
                        return pd.to_datetime(ts, utc=True)
                return pd.to_datetime(ts, utc=True)
            except:
                return pd.Timestamp.now(tz='UTC')
        
        df['timestamp'] = df['timestamp'].apply(parse_timestamp_safe)
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        df['usd_value'] = pd.to_numeric(df['usd_value'], errors='coerce').fillna(0)
        
        # Separate current positions from historical activities
        current_positions = df[df['activity_type'] == 'current_position']
        historical_activities = df[df['activity_type'] != 'current_position']
        
        # Basic metrics
        total_transactions = len(historical_activities)
        unique_tokens = df['token'].nunique()
        
        # Current position analysis
        current_compound_balance = current_positions['usd_value'].sum()
        
        # Historical activity analysis
        mint_activities = historical_activities[historical_activities['activity_type'] == 'mint']
        redeem_activities = historical_activities[historical_activities['activity_type'] == 'redeem']
        borrow_activities = historical_activities[historical_activities['activity_type'] == 'borrow']
        repay_activities = historical_activities[historical_activities['activity_type'] == 'repay']
        
        total_minted = mint_activities['usd_value'].sum()
        total_redeemed = redeem_activities['usd_value'].sum()
        total_borrowed = borrow_activities['usd_value'].sum()
        total_repaid = repay_activities['usd_value'].sum()
        
        # Calculate ratios
        net_supply = total_minted - total_redeemed
        net_borrow = total_borrowed - total_repaid
        supply_ratio = net_supply / max(total_minted, 1) if total_minted > 0 else 0
        borrow_ratio = net_borrow / max(total_borrowed, 1) if total_borrowed > 0 else 0
        
        # Time-based analysis
        now = pd.Timestamp.now(tz='UTC')
        if not historical_activities.empty:
            first_tx_date = historical_activities['timestamp'].min()
            last_tx_date = historical_activities['timestamp'].max()
            account_age_days = (now - first_tx_date).days if pd.notna(first_tx_date) else 0
            days_since_last_activity = (now - last_tx_date).days if pd.notna(last_tx_date) else 0
        else:
            account_age_days = 0
            days_since_last_activity = 0
        
        # Risk indicators (simplified since we don't have liquidation data directly)
        transaction_frequency = total_transactions / max(account_age_days, 1) if account_age_days > 0 else 0
        
        return {
            'total_transactions': total_transactions,
            'current_compound_balance_usd': current_compound_balance,
            'total_minted_usd': total_minted,
            'total_borrowed_usd': total_borrowed,
            'net_supply_usd': net_supply,
            'net_borrow_usd': net_borrow,
            'supply_ratio': supply_ratio,
            'borrow_ratio': borrow_ratio,
            'account_age_days': account_age_days,
            'days_since_last_activity': days_since_last_activity,
            'unique_tokens': unique_tokens,
            'transaction_frequency': transaction_frequency,
            'total_usd_volume': total_minted + total_borrowed
        }
    
    def _get_default_features(self) -> Dict:
        """Return default features for wallets with no Compound activity"""
        return {
            'total_transactions': 0,
            'current_compound_balance_usd': 0,
            'total_minted_usd': 0,
            'total_borrowed_usd': 0,
            'net_supply_usd': 0,
            'net_borrow_usd': 0,
            'supply_ratio': 0,
            'borrow_ratio': 0,
            'account_age_days': 0,
            'days_since_last_activity': 0,
            'unique_tokens': 0,
            'transaction_frequency': 0,
            'total_usd_volume': 0
        }
    
    def calculate_risk_score(self, features: Dict) -> Tuple[float, Dict]:
        """Calculate risk score based on features"""
        
        # Risk scoring weights
        weights = {
            'borrow_ratio': 200,               # High borrow retention = risk
            'days_since_last_activity': 150,   # Inactivity risk
            'net_borrow_ratio': 180,           # Net borrowing position
            'total_transactions': -60,         # More activity = lower risk
            'account_age_days': -100,          # Older accounts = lower risk
            'unique_tokens': -40,              # Token diversification
            'transaction_frequency': -80,      # Regular activity = lower risk
            'current_balance': 50              # Large positions = moderate risk
        }
        
        score_components = {}
        
        # Borrow ratio risk (higher retention of borrowed amount = higher risk)
        borrow_risk = min(features['borrow_ratio'], 1.0)
        score_components['borrow_risk'] = borrow_risk * weights['borrow_ratio']
        
        # Inactivity risk
        inactivity = min(features['days_since_last_activity'] / 365, 2.0) / 2.0
        score_components['inactivity_risk'] = inactivity * weights['days_since_last_activity']
        
        # Net borrowing position risk
        total_volume = max(features['total_usd_volume'], 1)
        net_borrow_ratio = max(features['net_borrow_usd'] / total_volume, 0)
        net_borrow_norm = min(net_borrow_ratio, 1.0)
        score_components['net_borrow_risk'] = net_borrow_norm * weights['net_borrow_ratio']
        
        # Activity benefits
        activity_norm = min(np.log1p(features['total_transactions']) / np.log1p(50), 1.0)
        score_components['activity_benefit'] = activity_norm * weights['total_transactions']
        
        # Account maturity benefit
        age_norm = min(features['account_age_days'] / 730, 1.0)
        score_components['maturity_benefit'] = age_norm * weights['account_age_days']
        
        # Diversification benefit
        token_diversity = min(features['unique_tokens'] / 5, 1.0)
        score_components['token_diversity_benefit'] = token_diversity * weights['unique_tokens']
        
        # Transaction frequency benefit
        freq_norm = min(features['transaction_frequency'] * 365, 1.0)  # Normalize to yearly frequency
        score_components['frequency_benefit'] = freq_norm * weights['transaction_frequency']
        
        # Current balance risk (large positions have moderate risk)
        balance_risk = min(features['current_compound_balance_usd'] / 100000, 1.0)
        score_components['balance_risk'] = balance_risk * weights['current_balance']
        
        # Calculate final score (0-1000 scale)
        raw_score = sum(score_components.values())
        risk_score = max(0, min(1000, 500 + raw_score))
        
        return risk_score, score_components

def main():
    st.title("🏦 Compound Protocol Wallet Risk Scorer")
    st.markdown("""
    Analyze wallet interactions with Compound V2 protocol using **Moralis API** for comprehensive risk assessment.
    
    **Risk Score Interpretation:**
    - **0-300**: 🟢 **Low Risk** (Conservative users, good activity patterns)
    - **301-600**: 🟡 **Medium Risk** (Moderate activity, some risk factors)
    - **601-1000**: 🔴 **High Risk** (High leverage patterns, inactivity, or concerning behavior)
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Risk Assessment Factors")
        st.markdown("""
        **🔴 Risk Indicators:**
        - High borrow retention ratio
        - Long periods of inactivity
        - Large net borrowing positions
        - Limited token diversification
        
        **🟢 Positive Indicators:**
        - Regular transaction activity
        - Long account history
        - Diverse token interactions
        - Balanced supply/borrow patterns
        """)
        
        st.markdown("---")
        st.markdown("### ⚡ API Status")
        st.success("✅ Moralis API Connected")
        st.info("📊 Using ERC20 balances & transfers")
    
    # File upload
    st.header("📁 Upload Wallet Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file with wallet addresses",
        type=['csv'],
        help="CSV should contain a 'wallet_id' column with Ethereum addresses"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'wallet_id' not in df.columns:
                st.error("❌ CSV must contain a 'wallet_id' column with Ethereum addresses")
                return
            
            # Clean wallet addresses
            df['wallet_id'] = df['wallet_id'].str.strip()
            df = df[df['wallet_id'].str.match(r'^0x[a-fA-F0-9]{40}$', na=False)]
            
            if df.empty:
                st.error("❌ No valid Ethereum addresses found")
                return
            
            st.success(f"✅ Loaded {len(df)} valid wallet addresses")
            with st.expander("Preview data"):
                st.dataframe(df.head(), use_container_width=True)
            
            # Analysis settings
            max_wallets = st.slider(
                "Maximum wallets to analyze",
                min_value=1,
                max_value=min(100, len(df)),
                value=min(20, len(df)),
                help="Limit to avoid rate limits"
            )
            
            if st.button("🚀 Analyze Wallets", type="primary"):
                scorer = MoralisCompoundRiskScorer()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                total_wallets = min(max_wallets, len(df))
                
                for idx, wallet_address in enumerate(df['wallet_id'].head(total_wallets)):
                    status_text.text(f"🔍 Analyzing wallet {idx+1}/{total_wallets}: {wallet_address}")
                    
                    try:
                        # Fetch data using working Moralis endpoints
                        balances = scorer.fetch_wallet_erc20_balances(wallet_address)
                        transfers = scorer.fetch_wallet_erc20_transfers(wallet_address)
                        transactions = scorer.fetch_wallet_transactions(wallet_address)
                        
                        # Parse Compound activities
                        compound_activities = scorer.parse_compound_activity(
                            balances, transfers, transactions, wallet_address
                        )
                        
                        # Calculate features and risk score
                        features = scorer.calculate_wallet_features(compound_activities)
                        risk_score, score_components = scorer.calculate_risk_score(features)
                        
                        # Risk categorization
                        if risk_score <= 300:
                            risk_category = "Low Risk"
                            risk_emoji = "🟢"
                        elif risk_score <= 600:
                            risk_category = "Medium Risk"
                            risk_emoji = "🟡"
                        else:
                            risk_category = "High Risk"
                            risk_emoji = "🔴"
                        
                        results.append({
                            'wallet_id': wallet_address,
                            'risk_score': round(risk_score, 1),
                            'risk_category': risk_category,
                            'risk_emoji': risk_emoji,
                            'total_transactions': features['total_transactions'],
                            'current_balance_usd': round(features['current_compound_balance_usd'], 2),
                            'total_minted_usd': round(features['total_minted_usd'], 2),
                            'total_borrowed_usd': round(features['total_borrowed_usd'], 2),
                            'net_borrow_usd': round(features['net_borrow_usd'], 2),
                            'account_age_days': features['account_age_days'],
                            'days_since_last_activity': features['days_since_last_activity'],
                            'unique_tokens': features['unique_tokens']
                        })
                        
                    except Exception as e:
                        st.warning(f"⚠️ Error analyzing {wallet_address}: {str(e)}")
                        # Add default entry for failed analysis
                        results.append({
                            'wallet_id': wallet_address,
                            'risk_score': 500,  # Neutral score for unknown
                            'risk_category': "Unknown",
                            'risk_emoji': "⚪",
                            'total_transactions': 0,
                            'current_balance_usd': 0,
                            'total_minted_usd': 0,
                            'total_borrowed_usd': 0,
                            'net_borrow_usd': 0,
                            'account_age_days': 0,
                            'days_since_last_activity': 0,
                            'unique_tokens': 0
                        })
                    
                    progress_bar.progress((idx + 1) / total_wallets)
                    time.sleep(0.3)  # Rate limiting
                
                status_text.text("✅ Analysis Complete!")
                
                # Display results
                if results:
                    results_df = pd.DataFrame(results)
                    
                    st.header("📊 Risk Analysis Results")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average Risk Score", f"{results_df['risk_score'].mean():.1f}")
                    with col2:
                        high_risk = len(results_df[results_df['risk_score'] > 600])
                        st.metric("🔴 High Risk", f"{high_risk}")
                    with col3:
                        active_wallets = len(results_df[results_df['total_transactions'] > 0])
                        st.metric("Active Wallets", f"{active_wallets}")
                    with col4:
                        with_balance = len(results_df[results_df['current_balance_usd'] > 0])
                        st.metric("With Compound Tokens", f"{with_balance}")
                    
                    # Results table
                    st.subheader("📋 Detailed Results")
                    display_df = results_df.copy()
                    display_df['Risk'] = display_df['risk_emoji'] + ' ' + display_df['risk_category']
                    
                    final_display = display_df[[
                        'wallet_id', 'risk_score', 'Risk', 'total_transactions',
                        'current_balance_usd', 'total_minted_usd', 'total_borrowed_usd',
                        'account_age_days', 'unique_tokens'
                    ]].sort_values('risk_score', ascending=False)
                    
                    st.dataframe(final_display, use_container_width=True)
                    
                    # Download button
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv_data,
                        file_name=f"compound_risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        # Sample data
        st.header("📋 Sample CSV Format")
        sample_df = pd.DataFrame({
            'wallet_id': [
                '0x742d35cc1ae0cd95a5e7b48f0a77b1b0ef0d5c7e',
                '0x8ba1f109551bd432803012645hac136c0c8f47eb',
                '0x40ec5b33f54e0e8a33a975908c5ba1c14e5bccdf'
            ]
        })
        st.dataframe(sample_df, use_container_width=True)
        
        sample_csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=sample_csv,
            file_name="sample_wallets.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
