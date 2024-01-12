import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates
import base64
from PIL import Image
from pandas.tseries.offsets import DateOffset

# Enable wide mode
st.set_page_config(layout="wide")

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
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_pivot, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(f"Weekly Activity Heatmap for Last {time_delta} Days")
    plt.xlabel("2-Hour Interval")
    plt.ylabel("Day of the Week")
    plt.xticks(rotation=45)
    st.pyplot()

def plot_bar_chart(data, title, xlabel, ylabel='Count'):
    """
    Plots a bar chart using the data provided.
    :param data: DataFrame with the data to plot
    :param title: Title of the plot
    :param xlabel: X-axis label
    :param ylabel: Y-axis label
    """
    # Create a new figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    data.plot(kind='bar', stacked=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(data.index, rotation=45)
    ax.legend()
    st.pyplot(fig)  # Pass the figure to st.pyplot() instead of relying on the current figure

def sentiment_analysis(df, period, period_title):
    """
    Performs sentiment analysis based on the period provided and plots a bar chart.
    :param df: DataFrame with the data
    :param period: A string specifying the period ('week', 'month', or '3month')
    :param period_title: A descriptive title for the period
    """
    # Generate a pivot table based on the period
    if period == 'week':
        df['day_of_week'] = df['timestamp'].dt.day_name()
        pivot_data = df.pivot_table(index='day_of_week', columns='sentiment', values='timestamp', aggfunc='count').reindex([
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    elif period == 'month':
        df['week_of_month'] = df['timestamp'].apply(lambda x: (x.day - 1) // 7 + 1)
        pivot_data = df.pivot_table(index='week_of_month', columns='sentiment', values='timestamp', aggfunc='count')
    elif period == '3month':
        df['month_year'] = df['timestamp'].dt.to_period('M')
        pivot_data = df.pivot_table(index='month_year', columns='sentiment', values='timestamp', aggfunc='count').iloc[-3:]

    # Plot the bar chart
    plot_bar_chart(pivot_data, f'Sentiment Analysis by {period_title}', period_title)

# Utility function to get start date based on time period
def get_start_date(end_date, period):
    if period == 'week':
        return end_date - DateOffset(days=7)
    elif period == 'month':
        return end_date - DateOffset(months=1)
    elif period == '3months':
        return end_date - DateOffset(months=3)
    else:
        return None

# Refactored function for line graph
def line_graph(df, period):
    end_date = df['timestamp'].max()
    start_date = get_start_date(end_date, period)

    if period == 'week':
        period_data = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        counts = period_data.groupby(period_data['timestamp'].dt.date).size()
        x_labels = counts.index.strftime('%A')
    elif period == 'month':
        period_data = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        period_data['week_of_month'] = period_data['timestamp'].apply(lambda x: (x.day - 1) // 7 + 1)
        counts = period_data.groupby('week_of_month').size()
        x_labels = range(1, 6)
    elif period == '3months':
        period_data = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        counts = period_data.groupby(period_data['timestamp'].dt.to_period('M')).size()
        x_labels = counts.index.astype(str)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=counts.index, y=counts.values, marker='o')
    plt.title(f'Queries in the Latest {period.capitalize()}')
    plt.xlabel('Time Period')
    plt.ylabel('Number of Queries')
    plt.xticks(ticks=counts.index, labels=x_labels, rotation=45)
    plt.grid(True)
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
    with col2:
        if time_delta_option == "1 week":
            sentiment_analysis(df, 'week', 'Day of the Week')
        elif time_delta_option == "1 month":
            sentiment_analysis(df, 'month', 'Week of the Month')
        elif time_delta_option == "3 months":
            sentiment_analysis(df, '3month', 'Month')
    with col3:
        if time_delta_option == "1 week":
            line_graph(df, 'week')
        elif time_delta_option == "1 month":
            line_graph(df, 'month')
        elif time_delta_option == "3 months":
            line_graph(df, '3months')

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def set_background(image_file):
    bin_str = get_base64_encoded_image(image_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
# Call the function to add the background
set_background('./source/background.jpg')

# Initialize session state
if 'current_tab' not in st.session_state:
    st.session_state['current_tab'] = 'Dashboard'

# Load your company logo
logo = Image.open('./source/devan&company.png')

# Main
main_layout()