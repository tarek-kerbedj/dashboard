import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates
from PIL import Image
from pandas.tseries.offsets import DateOffset
from datetime import timedelta

# Load data from CSV
df = pd.read_csv('./data/chat_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Define function to create heatmap for given time delta
def create_heatmap(df, time_delta):
    # Filter data for the specified time period
    end_date = df['timestamp'].max()
    start_date = end_date - pd.Timedelta(days=time_delta)
    filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

    # Extract day of the week and 2-hour intervals
    filtered_df['day_of_week'] = filtered_df['timestamp'].dt.day_name()
    filtered_df['hour'] = filtered_df['timestamp'].dt.hour
    filtered_df['hour_group'] = pd.cut(filtered_df['hour'], bins=np.arange(0, 25, 2), right=False, labels=[f"{i}-{i+2}" for i in range(0, 24, 2)])

    # Set order for days of the week
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    filtered_df['day_of_week'] = pd.Categorical(filtered_df['day_of_week'], categories=ordered_days, ordered=True)

    # Pivot the DataFrame
    df_pivot = filtered_df.pivot_table(index='day_of_week', columns='hour_group', values='timestamp', aggfunc='count', fill_value=0)

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_pivot, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(f"Weekly Activity Heatmap for Last {time_delta} Days")
    plt.xlabel("2-Hour Interval")
    plt.ylabel("Day of the Week")
    plt.xticks(rotation=45)
    plt.show()

# Function to plot weekly sentiment analysis
def plot_weekly_sentiment_analysis(df, time_delta):
    # Weekly Sentiment Analysis
    latest_date = df['timestamp'].max()
    start_date = latest_date - timedelta(days=time_delta)
    last_week_data = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= latest_date)]
    last_week_data['day_of_week'] = last_week_data['timestamp'].dt.day_name()
    sentiment_counts_week = last_week_data.groupby(['day_of_week', 'sentiment']).size().unstack()
    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sentiment_counts_week = sentiment_counts_week.reindex(ordered_days).fillna(0)
    plot_bar_chart(sentiment_counts_week, "Days of the Week")

# Function to plot monthly sentiment analysis
def plot_monthly_sentiment_analysis(df):
    # Monthly Sentiment Analysis
    latest_date = df['timestamp'].max()
    start_month_date = latest_date - pd.DateOffset(months=1)
    last_month_data = df[(df['timestamp'] >= start_month_date) & (df['timestamp'] <= latest_date)]
    last_month_data['week_of_month'] = last_month_data['timestamp'].apply(lambda x: (x.day - 1) // 7 + 1)
    sentiment_counts_month = last_month_data.groupby(['week_of_month', 'sentiment']).size().unstack()
    all_weeks = range(1, 6)
    sentiment_counts_month = sentiment_counts_month.reindex(all_weeks).fillna(0)
    plot_bar_chart(sentiment_counts_month, "Weeks of the Month")

# Function to plot 3 months sentiment analysis
def plot_3_months_sentiment_analysis(df):
    # 3 Months Sentiment Analysis
    df['month_year'] = df['timestamp'].dt.to_period('M')
    sentiment_counts_3months = df.groupby(['month_year', 'sentiment']).size().unstack().iloc[-3:]
    plot_bar_chart(sentiment_counts_3months, "Month")

# Function for line graph of the latest week with days of the week
def line_graph_latest_week(df):
    end_date = df['timestamp'].max()
    start_date = end_date - DateOffset(days=7)
    week_data = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    daily_counts = week_data.groupby(week_data['timestamp'].dt.date).size()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=daily_counts.index, y=daily_counts.values, marker='o')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%A'))
    plt.title('Queries in the Latest Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Queries')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot()

# Function for line graph of the latest month (weeks)
def line_graph_latest_month(df):
    end_date = df['timestamp'].max()
    start_date = end_date - DateOffset(months=1)
    month_data = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    month_data['week_of_month'] = month_data['timestamp'].apply(lambda x: (x.day - 1) // 7 + 1)
    weekly_counts = month_data.groupby('week_of_month').size().reindex(range(1, 6))

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=weekly_counts.index, y=weekly_counts.values, marker='o')
    plt.title('Queries in the Latest Month by Week')
    plt.xlabel('Week of Month')
    plt.ylabel('Number of Queries')
    plt.xticks(range(1, 6))
    plt.grid(True)
    st.pyplot()

# Function for line graph of the latest 3 months (months)
def line_graph_latest_3_months(df):
    end_date = df['timestamp'].max()
    start_date = end_date - DateOffset(months=3)
    months_data = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    monthly_counts = months_data.groupby(months_data['timestamp'].dt.to_period('M')).size()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=monthly_counts.index.astype(str), y=monthly_counts.values, marker='o')
    plt.title('Queries in the Latest 3 Months')
    plt.xlabel('Month')
    plt.ylabel('Number of Queries')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot()

# Function to plot bar chart
def plot_bar_chart(data, xlabel):
    plt.figure(figsize=(10, 6))
    bar_width = 0.2
    index = np.arange(len(data))

    for i, sentiment in enumerate(data.columns):
        plt.bar(index + i * bar_width, data[sentiment], bar_width, label=f'{sentiment.capitalize()}')

    plt.xlabel(xlabel)
    plt.ylabel('Percentage')
    plt.xticks(index + bar_width, data.index)
    plt.legend()
    plt.tight_layout()
    st.pyplot()

# UI Layout
def main_layout():
    with st.sidebar:
        st.image(logo, width=300)
        st.title("Navigation")
        st.markdown("---")

        if st.button("📊 Dashboard"):
            st.session_state['current_tab'] = 'Dashboard'
        if st.button("💬 Conversation"):
            st.session_state['current_tab'] = 'Conversation'

        st.markdown("---")
        st.markdown("## About")
        st.info("This app provides an interactive analysis of chat data.")

    if st.session_state['current_tab'] == 'Conversation':
        conversation_tab()
    elif st.session_state['current_tab'] == 'Dashboard':
        dashboard_tab()

def conversation_tab():
    st.subheader("Chat")
    st.write(df.sort_values(by='timestamp', ascending=False))

def dashboard_tab():
    st.subheader("Dashboard")
    time_delta_option = st.selectbox("Select Time Period", ["1 week", "1 month", "3 months"])
    time_delta = {"1 week": 7, "1 month": 30, "3 months": 90}[time_delta_option]

    col1, col2, col3 = st.columns(3)
    with col1:
        create_heatmap(df, time_delta)
        st.pyplot()
    with col2:
        plot_weekly_sentiment_analysis(df, time_delta) if time_delta_option == "1 week" else (plot_monthly_sentiment_analysis(df) if time_delta_option == "1 month" else plot_3_months_sentiment_analysis(df))
    with col3:
        line_graph_latest_week(df) if time_delta_option == "1 week" else (line_graph_latest_month(df) if time_delta_option == "1 month" else line_graph_latest_3_months(df))



# Initialize session state
if 'current_tab' not in st.session_state:
    st.session_state['current_tab'] = 'Dashboard'

# Load your company logo
logo = Image.open('./devan&company.png')

# Main
main_layout()