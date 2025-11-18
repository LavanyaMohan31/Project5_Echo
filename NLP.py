import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.figure_factory as ff
import joblib
import openpyxl
from imblearn.pipeline import Pipeline as IMBPipeline
from wordcloud import WordCloud
#--------------------------------------------------------------------------------------------------------------
# Load Data
df=pd.read_excel("C:/DS_Programs/Project5_ECHO/AI_Echo.xlsx")
#--------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="AI ECHO",page_icon="🤖", layout="wide")

#sidebar
def show_tab(tab_name):
    st.session_state["active_tab"] = tab_name
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Home"

# Custom CSS for equal-size buttons + text color
st.markdown("""
    <style>
    /* Center buttons inside sidebar */
    .stSidebar .element-container {
        display: flex;
        justify-content: center;
    }

    /* Style buttons */
    div.stButton > button {
        width: 170px;            /* Equal width */
        height: 50px;            /* Equal height */
        font-size: 24px;           /* Font size */
        color: white !important; /* Text color */
        background-color: #9e0255; /* Background color */
        border-radius: 8px;      /* Rounded corners */
        font-weight: bold;
        margin: 2px 0;          /* Space between buttons */
    }
    div.stButton > button:hover {
        background-color: #d13488; /* Hover effect */
        color: #fff !important;
    }

    /* Add top margin to push buttons down */
    .stSidebar .stButton:first-child {
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar buttons
with st.sidebar:
    st.button("🏠 Home", on_click=show_tab, args=("Home",))
    st.button("📊 Dashboard", on_click=show_tab, args=("Dashboard",))
    st.button("🔍 Prediction", on_click=show_tab, args=("Prediction",))

# Lock sidebar width using CSS
st.markdown("""
    <style>
    /* Fix the sidebar width */
    section[data-testid="stSidebar"] {
        width: 200px !important;
        min-width: 200px !important;
        max-width: 200px !important;
        /*background-color: #e6f2ff;*/
        background: linear-gradient(#ed98c5,#ed98c5,#db6ea8, #d13488, #9e0255); /* Blue gradient */
        color: white; /* Text color */
    }
            
    /* Ensure all text inside sidebar inherits the color */
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    /* Fix the main content position accordingly */
    section.main {
        margin-left: 200px;
    }
  
    </style>
""", unsafe_allow_html=True)




# Sticky title with space for the sidebar

st.markdown("""
    <style>
    /* Fixed header below top navbar & beside sidebar */
    .fixed-title {
        position: fixed;
        top: 3.5rem;  /* Adjusted for Streamlit's top nav (approx 56px) */
        left: 12.5rem;  /* Sidebar width */
        width: calc(100% - 14rem);
        z-index: 9999;
        background-color: #9c025a;
        color: white;
        text-align: center;
        padding: 10px 0;
        font-size: 24px;
        font-weight: bold;
        border-bottom: 2px solid #52022f;
    }

    /* Push content below fixed header */
    .spacer {
        height: 5px;
    }
    </style>

    <div class="fixed-title">
        🔊 AI ECHO - Your Smartest Conversational Partner
    </div>
    <div class="spacer"></div>
""", unsafe_allow_html=True)

#---------------------------------------------------------------------------------------------
#Styled Header
def styled_header(text, icon="📊"):
    st.markdown(f"""
        <style>
        .styled-h3 {{
            background-color: #9c025a !important;
            color: white !important;
            border: 2px solid #52022f;
            padding: 4px 10px !important;
            border-radius: 5px;
            text-align: left;
            font-size: 18px !important;
            font-weight: 600 !important;
            margin-bottom: 10px !important;
            line-height: 1.2 !important;
            width: fit-content;
            margin-left: auto;
            margin-right: auto;
        }}
        </style>

        <h3 class="styled-h3">{icon} {text}</h3>
    """, unsafe_allow_html=True)
#-------------------------------------------------------------------------------------------
def styled_title(text, color="#d13488", size="24px", align="center"):
    st.markdown(
        f"""
        <h2 style='text-align: {align}; 
                   color: {color}; 
                   font-size: {size}; 
                   font-weight: bold;
                   text-shadow: 1px 1px 2px #fff;'>
            {text}
        </h2>
        """,
        unsafe_allow_html=True
    )
#---------------------------------------------------------------------------------------------
# Show selected tab content
if st.session_state["active_tab"] == "Home":
    st.markdown("""
        <h1 style='text-align: center; color: #9e0255; font-family: "Arial", sans-serif;'>
            Welcome
        </h1>
    """, unsafe_allow_html=True)

  
    st.image(
        "C:/DS_Programs/PROJECT5_ECHO/img.png",  # replace with your image path or URL
         use_container_width=True
    )

#----------------------------------------------------------------------------------------------------------------
elif st.session_state["active_tab"] == "Dashboard":
    st.markdown("""
        <h1 style='text-align: center; color: #9e0255; font-family: "Arial", sans-serif;'>
            Dashboard
        </h1>
    """, unsafe_allow_html=True)
       #**************************************************************
    styled_header("Sentimental Analysis", icon="📊")
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    styled_title("Overall sentiment of user reviews")
    sentiment_counts = (
        df['sentiment_label']
        .value_counts(normalize=True)
        .mul(100)
        .reset_index()
    )
    sentiment_counts.columns = ['Sentiment', 'Percentage']

    # Plotly pie chart
    fig = px.pie(
        sentiment_counts,
        names='Sentiment',
        values='Percentage',
        color='Sentiment',
        color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'},
        title='Overall Sentiment Distribution',
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')

    # Display chart
    st.plotly_chart(fig, use_container_width=True)

    # Show summary metrics
    col1, col2, col3 = st.columns(3)

    if 'Positive' in sentiment_counts['Sentiment'].values:
        col1.metric("😊 Positive", f"{sentiment_counts[sentiment_counts['Sentiment']=='Positive']['Percentage'].values[0]:.1f}%")
    else:
        col1.metric("😊 Positive", "0.0%")

    if 'Neutral' in sentiment_counts['Sentiment'].values:
        col2.metric("😐 Neutral", f"{sentiment_counts[sentiment_counts['Sentiment']=='Neutral']['Percentage'].values[0]:.1f}%")
    else:
        col2.metric("😐 Neutral", "0.0%")

    if 'Negative' in sentiment_counts['Sentiment'].values:
        col3.metric("😠 Negative", f"{sentiment_counts[sentiment_counts['Sentiment']=='Negative']['Percentage'].values[0]:.1f}%")
    else:
        col3.metric("😠 Negative", "0.0%")
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    styled_title("Sentiment vary by rating")
    rating_sentiment = df.groupby(['rating', 'sentiment_label']).size().reset_index(name='count')

    rating_sentiment['percentage'] = (
        rating_sentiment.groupby('rating')['count']
        .transform(lambda x: round(100 * x / x.sum(), 2))
    )

    # Create stacked bar chart
    fig = px.bar(
        rating_sentiment,
        x='rating',
        y='percentage',
        color='sentiment_label',
        color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'},
        title="",
        labels={'rating': 'User Rating (1–5)', 'percentage': 'Percentage', 'sentiment_label': 'Sentiment'}
    )

    fig.update_layout(
        xaxis=dict(dtick=1),
        yaxis_title='Percentage (%)',
        plot_bgcolor='white',
        title_x=0.5,
        legend_title_text='Sentiment',
        bargap=0.15
    )

    st.plotly_chart(fig, use_container_width=True)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    styled_title("Keywords associated with each sentiment class")
    from collections import Counter

    df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip().str.lower()
    # --- WORDCLOUD FUNCTION ---
    def generate_wc(text):
        if len(text.strip()) == 0:
            return None
        return WordCloud(width=600, height=400, background_color='white').generate(text)

    # --- KEYWORD TABLE FUNCTION ---
    def get_top_keywords(text_list, top_n=20):
        if len(text_list) == 0:
            return pd.DataFrame({"keyword": [], "frequency": []})
        words = " ".join(text_list).split()
        freq = Counter(words).most_common(top_n)
        return pd.DataFrame(freq, columns=["keyword", "frequency"])

    pos_texts = df[df["sentiment_label"] == "positive"]["Final_text"].tolist()
    neg_texts = df[df["sentiment_label"] == "negative"]["Final_text"].tolist()
    neu_texts = df[df["sentiment_label"] == "neutral"]["Final_text"].tolist()

    pos_text = " ".join(pos_texts)
    neg_text = " ".join(neg_texts)
    neu_text = " ".join(neu_texts)
             #------------------------------------------
    st.subheader("Sentiment WordClouds and Keyword Frequency Table")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Positive")
        wc = generate_wc(pos_text)
        if wc:
            fig, ax = plt.subplots(figsize=(4,4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.warning("No positive reviews.")
        st.dataframe(get_top_keywords(pos_texts))

    with col2:
        st.subheader("Negative")
        wc = generate_wc(neg_text)
        if wc:
            fig, ax = plt.subplots(figsize=(4,4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.warning("No negative reviews.")
        st.dataframe(get_top_keywords(neg_texts))

    with col3:
        st.subheader("Neutral")
        wc = generate_wc(neu_text)
        if wc:
            fig, ax = plt.subplots(figsize=(4,4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.warning("No neutral reviews.")
        st.dataframe(get_top_keywords(neu_texts))
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    styled_title("Verified Users Sentiment Comparison")

    df["verified_purchase"] = df["verified_purchase"].astype(str).str.strip().str.lower()
    df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip().str.lower()

    verified_sentiment = (
        df.groupby(["verified_purchase", "sentiment_label"])
        .size()
        .reset_index(name="count")
    )

    verified_sentiment["percentage"] = (
        verified_sentiment.groupby("verified_purchase")["count"]
        .transform(lambda x: 100 * x / x.sum())
    )

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(
        data=verified_sentiment,
        x="verified_purchase",
        y="percentage",
        hue="sentiment_label",
        ax=ax
    )
    ax.set_title("Sentiment Distribution by Verified Purchase")
    ax.set_xlabel("Verified Purchase")
    ax.set_ylabel("Percentage (%)")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    styled_title("Average Review Length by Sentiment")
    df["review_length"] = df["review"].astype(str).apply(lambda x: len(x.split()))

    length_stats = df.groupby("sentiment_label")["review_length"].mean()
    st.write(length_stats)
    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column="review_length", by="sentiment_label", ax=ax)
    plt.title("Review Length vs Sentiment")
    plt.suptitle("")
    plt.xlabel("Sentiment")
    plt.ylabel("Review Length (word count)")
    st.pyplot(fig)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    styled_title("Location-wise Sentiment Analysis")

    # Ensure location column is clean
    df["location"] = df["location"].astype(str).str.strip().str.title()

    # Sentiment counts for every location
    location_sentiment = df.groupby(["location", "sentiment_label"]).size().reset_index(name="count")

    top_positive = (
        location_sentiment[location_sentiment["sentiment_label"] == "positive"]
        .sort_values("count", ascending=False)
        .head(5)
    )

    top_negative = (
        location_sentiment[location_sentiment["sentiment_label"] == "negative"]
        .sort_values("count", ascending=False)
        .head(5)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Top 10 Positive Locations")
        st.dataframe(top_positive)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(top_positive["location"], top_positive["count"])
        ax.set_title("Top 10 Locations with Positive Reviews")
        ax.set_xlabel("Count")
        ax.set_ylabel("Location")
        st.pyplot(fig)

    with col2:
        st.write("### Top 10 Negative Locations")
        st.dataframe(top_negative)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(top_negative["location"], top_negative["count"])
        ax.set_title("Top 10 Locations with Negative Reviews")
        ax.set_xlabel("Count")
        ax.set_ylabel("Location")
        st.pyplot(fig)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    styled_title("Sentiment Comparison Across Platforms (Web vs Mobile)")
    df["platform"] = df["platform"].astype(str).str.strip().str.title()
    df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip().str.lower()

    platform_sentiment = (
    df.groupby(["platform", "sentiment_label"])
      .size()
      .reset_index(name="count")
    )

    plt.figure(figsize=(10,5))
    sns.barplot(
        data=platform_sentiment,
        x="platform",
        y="count",
        hue="sentiment_label"
    )
    plt.title(" ")
    plt.xlabel("Platform")
    plt.ylabel("Number of Reviews")
    st.pyplot(plt)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    styled_title("Version Sentiment Score ")
    df["version"] = df["version"].astype(str).str.strip()
    df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip().str.lower()

    version_score = (
    df.replace({"sentiment_label": {"positive": 1, "neutral": 0, "negative": -1}})
      .groupby("version")["sentiment_label"]
      .mean()
      .reset_index(name="sentiment_score")
      .sort_values("sentiment_score", ascending=False)
    )
    st.dataframe(version_score)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    styled_title("Most Common Negative Feedback Themes")

    neg_df = df[df["sentiment_label"] == "negative"]

    if neg_df.empty:
        st.warning("No negative reviews found.")
    else:
        neg_text = " ".join(neg_df["Final_text"].astype(str))

        words = neg_text.split()

        # Frequency count
        from collections import Counter
        word_freq = Counter(words).most_common(30)

        st.write("### 🔥 Top 30 Frequent Negative Keywords")
        st.dataframe(pd.DataFrame(word_freq, columns=["word", "count"]))

        themes = {
            "Performance / Speed": ["slow", "lag", "freeze", "loading", "delay"],
            "Accuracy Issues": ["wrong", "error", "incorrect", "bad", "mistake"],
            "UI / Usability": ["difficult", "confusing", "interface", "navigation"],
            "Missing Features": ["missing", "lack", "cannot", "unable"],
            "Pricing": ["expensive", "price", "cost", "subscription", "payment"],
            "Customer Support": ["support", "help", "service"],
            "Bugs / Crashes": ["bug", "crash", "issue", "problem", "fail"]
        }

        theme_results = {}

        for theme, i in themes.items():
            count = sum(1 for w in words if w in i)
            theme_results[theme] = count

        theme_df = pd.DataFrame(theme_results.items(), columns=["Theme", "Count"])
        theme_df = theme_df.sort_values("Count", ascending=False)
        #st.dataframe(theme_df)
        st.bar_chart(theme_df.set_index("Theme"))
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#--------------------------------------------------------------------------------------------------
elif st.session_state["active_tab"] == "Prediction":
    styled_header("SENTIMENT PREDICTION")
    df1=pd.read_csv("C:/DS_Programs/Project5_ECHO/ChatGPT_Review.csv")
    model = joblib.load("C:/DS_Programs/Project5_ECHO/NLP_model.pkl")  

    def predict_sentiment(text):
        pred = model.predict([text])[0]    # model expects a list
        return pred
    user_input = st.text_area("✍️ Type your review here")
    
    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a review.")
        else:
            result = predict_sentiment(user_input)
            if result.lower() == "positive":
                st.success(f"🎉 Predicted Sentiment: **{result.upper()}**")
            else:
                st.error(f"❗ Predicted Sentiment: **{result.upper()}**")
