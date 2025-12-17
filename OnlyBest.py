# Run from command bar
# streamlit run "/home/zabaione/Documents/Volaterra/EDU/_Taleb, MIT Derivatives + Quant/amazon_purchase_app.py"
# streamlit run "C:/Users/lilip/Documents/Volaterra/EDU/_Taleb, MIT Derivatives + Quant/amazon_purchase_app.py"
# see on http://localhost:8501
# there is a deploy button?



import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Amazon Product Comparator", page_icon="🛒", layout="wide")

st.title("🛒 Amazon Product Comparator")
st.markdown(
    """
    **Compare two Amazon products** based on price and review quality using Bayesian inference.
    
    - **Inputs**: Price, number of 5⭐ and 4⭐ reviews, total reviews.
    - **Model**: Treats 5⭐ + 4⭐ as "successes" (quality probability *p*), rest as "failures".
    - **Bayesian Posterior**: Beta(1 + successes, 1 + failures) — uniform prior.
    - **Value**: *p* / price (bang for buck).
    - **Output**: Probability one product gives **higher value** via 100,000 Monte Carlo simulations.
    """
)

# Sidebar with explanation
with st.sidebar:
    st.header("📊 How it Works")
    st.markdown("""
    **Beta Distribution for Reviews**  
    The Beta distribution models the true quality probability *p* (chance of 4+⭐ review).  
    - **Prior**: Uniform Beta(1,1) — no bias.  
    - **Posterior**: Beta(1 + successes, 1 + failures).  
    - **Simulation**: Sample *p₁* and *p₂*, compute values *p/price*, compare *P(value₁ > value₂)*.  
    
    **Why Bayesian?** Handles uncertainty perfectly, especially low reviews.  
    **Example**: Product with few reviews has **wide** posterior → more risk.
    """)
    st.markdown("---")
    st.info("💡 **Deployed live**: [Try on Streamlit Cloud](#)")

# Inputs
col1, col2 = st.columns(2)

with col1:
    st.header("📦 Product 1")
    price1 = st.number_input("**Price** (€)", min_value=0.01, value=209.00, step=0.01, format="%.2f")
    five1 = st.number_input("**5⭐ Reviews**", min_value=0, value=1000, step=1)
    four1 = st.number_input("**4⭐ Reviews**", min_value=0, value=182, step=1)
    total1 = st.number_input("**Total Reviews**", min_value=1, value=1407, step=1)

with col2:
    st.header("📦 Product 2")
    price2 = st.number_input("**Price** (€)", min_value=0.01, value=179.00, step=0.01, format="%.2f")
    five2 = st.number_input("**5⭐ Reviews**", min_value=0, value=95, step=1)
    four2 = st.number_input("**4⭐ Reviews**", min_value=0, value=15, step=1)
    total2 = st.number_input("**Total Reviews**", min_value=1, value=125, step=1)

# Validation
if st.button("🚀 **Run Comparison**", type="primary", use_container_width=True):
    successes1 = five1 + four1
    successes2 = five2 + four2
    
    if successes1 > total1 or successes2 > total2:
        st.error("❌ **5⭐ + 4⭐ cannot exceed total reviews!**")
        st.stop()
    
    failures1 = total1 - successes1
    failures2 = total2 - successes2
    
    a1, b1 = 1 + successes1, 1 + failures1
    a2, b2 = 1 + successes2, 1 + failures2
    
    # Simulation
    N = 100_000
    progress_bar = st.progress(0)
    status = st.empty()
    
    status.text("🔄 Simulating posteriors...")
    p1 = np.random.beta(a1, b1, N)
    p2 = np.random.beta(a2, b2, N)
    progress_bar.progress(30)
    
    status.text("⚡ Computing values...")
    value1 = p1 / price1
    value2 = p2 / price2
    progress_bar.progress(70)
    
    prob1_better = np.mean(value1 > value2)
    prob2_better = np.mean(value2 > value1)
    prob_tie = 1 - prob1_better - prob2_better
    progress_bar.progress(100)
    
    # Results
    st.success("✅ **Simulation complete!**")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("**Product 1 Wins**", f"{prob1_better:.1%}", delta=f"{prob1_better - 0.5:+.1%}")
    with col_b:
        st.metric("**Product 2 Wins**", f"{prob2_better:.1%}", delta=f"{prob2_better - 0.5:+.1%}")
    with col_c:
        st.metric("**Tie**", f"{prob_tie:.3%}")
    
    # Comparative Table
    st.subheader("📈 **Detailed Comparison**")
    summary_df = pd.DataFrame({
        "Metric": [
            "Probability Better",
            "Mean Quality (*p*)",
            "Mean Value (*p*/price)",
            "95% Value Interval",
            "Successes / Failures"
        ],
        "Product 1": [
            f"{prob1_better:.1%}",
            f"{np.mean(p1):.1%}",
            f"{np.mean(value1):.4f}",
            f"{np.percentile(value1, 2.5):.4f} – {np.percentile(value1, 97.5):.4f}",
            f"{successes1} / {failures1}"
        ],
        "Product 2": [
            f"{prob2_better:.1%}",
            f"{np.mean(p2):.1%}",
            f"{np.mean(value2):.4f}",
            f"{np.percentile(value2, 2.5):.4f} – {np.percentile(value2, 97.5):.4f}",
            f"{successes2} / {failures2}"
        ]
    })
    st.table(summary_df)
    
    # Visuals
    st.subheader("🎨 **Visual Comparison**")
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        fig_hist1 = px.histogram(
            x=value1, nbins=50, title="**Product 1: Value Distribution**",
            labels={"x": "Value (*p*/price)"}
        )
        fig_hist1.update_layout(height=300)
        st.plotly_chart(fig_hist1, use_container_width=True)
    
    with col_v2:
        fig_hist2 = px.histogram(
            x=value2, nbins=50, title="**Product 2: Value Distribution**",
            labels={"x": "Value (*p*/price)"}
        )
        fig_hist2.update_layout(height=300)
        st.plotly_chart(fig_hist2, use_container_width=True)
    
    # Winner
    st.markdown("---")
    if prob1_better > prob2_better:
        st.balloons()
        st.markdown("🏆 **Recommendation: Product 1**")
    elif prob2_better > prob1_better:
        st.balloons()
        st.markdown("🏆 **Recommendation: Product 2**")
    else:
        st.warning("🤝 **Indifferent — flip a coin!**")

# Footer
st.markdown("---")
st.markdown(
    "<small>Made with ❤️ using Streamlit | Bayesian magic for smarter shopping</small>",
    unsafe_allow_html=True
)