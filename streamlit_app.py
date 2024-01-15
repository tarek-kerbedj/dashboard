import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64
from datetime import timedelta
from math import pi
from PIL import Image
from pandas.tseries.offsets import DateOffset

st.set_page_config(layout="wide")

# Load data from CSV
df = pd.read_csv('./data/chat_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

def create_heatmap(df, time_delta, title):
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
    plt.title(title)
    plt.xlabel("2-Hour Interval")
    plt.ylabel("Day of the Week")
    plt.xticks(rotation=45)
    st.pyplot()

def sentiment_analysis(df, period, period_title):
    """
    Performs sentiment analysis based on the period provided and plots a radar chart.
    :param df: DataFrame with the data
    :param period: A string specifying the period ('week', 'month', or '3month')
    :param period_title: A descriptive title for the period
    """
    # Get the most recent date in your dataset
    end_date = df['timestamp'].max()

    # Define the start_date based on the period
    if period == 'week':
        start_date = end_date - pd.Timedelta(days=6)
    elif period == 'month':
        start_date = end_date - pd.Timedelta(days=30)
    elif period == '3month':
        start_date = end_date - pd.Timedelta(days=90)

    # Filter the DataFrame for the selected period
    filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

    # Aggregate data
    sentiment_counts = filtered_df.groupby('sentiment')['timestamp'].count()
    avg_sentiment_counts = sentiment_counts / len(filtered_df['timestamp'].unique())

    # Convert to a format suitable for radar chart
    categories = list(avg_sentiment_counts.index)
    N = len(categories)

    # Calculate the angles for the radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Radar chart setup
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Values for radar chart
    values = avg_sentiment_counts.tolist()
    values += values[:1]  # Repeat the first value to close the circle

    # Draw the radar chart
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, alpha=0.2)

    # Add labels with emojis for each sentiment
    emoji_dict = {
        "Positive": "😃",
        "Neutral": "😐",
        "Negative": "😢",
    }
    labels = [emoji_dict.get(cat, cat) for cat in categories]

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)  # Use labels with emojis

    # Add title
    plt.title(f'Average Sentiment Analysis ({period_title})')

    # Display the chart
    st.pyplot(fig)
    
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

def users_queries_line_graph(df, period):
    end_date = df['timestamp'].max()

    if period == 'week':
        start_date = end_date - pd.DateOffset(days=6)
    elif period == 'month':
        start_date = end_date - pd.DateOffset(weeks=3)
    elif period == '3months':
        end_date_month = end_date.replace(day=1)
        start_date = end_date_month - pd.DateOffset(months=2)
    else:
        return None

    period_data = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

    if period == 'week':
        y_labels = period_data['timestamp'].dt.strftime('%A').unique()
        queries_per_period = period_data.groupby(period_data['timestamp'].dt.date).size()
        users_per_period = period_data.groupby(period_data['timestamp'].dt.date)['user_id'].nunique()
    elif period == 'month':
        # Calculate the week of the month
        period_data['week_of_month'] = period_data['timestamp'].apply(lambda x: (x.day - 1) // 7 + 1)
        y_labels = ["Week1", "Week2", "Week3", "Week4"]
        queries_per_period = period_data.groupby('week_of_month').size()
        users_per_period = period_data.groupby('week_of_month')['user_id'].nunique()
    elif period == '3months':
        y_labels = period_data['timestamp'].dt.strftime('%Y-%m').unique()
        queries_per_period = period_data.groupby(period_data['timestamp'].dt.to_period('M')).size()
        users_per_period = period_data.groupby(period_data['timestamp'].dt.to_period('M'))['user_id'].nunique()

    plt.figure(figsize=(10, 6))

    # Plotting the line graphs or scatter plots
    plt.plot(y_labels, queries_per_period, label='Number of Queries', color='blue', marker='o')
    plt.plot(y_labels, users_per_period, label='Number of Users', color='red', marker='o')

    # Adding annotations (optional)
    for label, qp, up in zip(y_labels, queries_per_period, users_per_period):
        plt.annotate(f'{qp}', (label, qp), textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f'{up}', (label, up), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title(f'Queries and Users in the Latest {period.capitalize()}')
    plt.xlabel('Time Period')
    plt.ylabel('Count')

    plt.legend(loc='upper left', bbox_to_anchor=(0.7, 1.0))
    plt.grid(True)
    st.pyplot()

def plot_error_types_distribution(df, time_period):
    # Set time period start and end dates
    end_date = df['timestamp'].max().date()
    
    if time_period == '1W':
        start_date = end_date - pd.DateOffset(days=7)
    elif time_period == '1M':
        start_date = end_date - pd.DateOffset(weeks=4)  # Adjusted to 4 weeks
    elif time_period == '3M':
        # Calculate the end date for the 3-month period
        start_date = end_date - pd.DateOffset(weeks=12)  # 12 weeks for 3 months
        start_date = start_date.replace(day=1)  # Start from the 1st day of the month

    # Filter data for the specified time period
    time_period_data = df[(df['timestamp'] >= pd.Timestamp(start_date)) & (df['timestamp'] <= pd.Timestamp(end_date))]

    # Determine the appropriate grouping and title based on time period
    if time_period == '1W':
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        grouping_column = time_period_data['timestamp'].dt.strftime('%A')
        title = 'Error Types Distribution Over the Last 7 Days'
        xlabel = 'Day of the Week'
        ylabel = 'Count'
    elif time_period == '1M':
        # Calculate week number within the month
        time_period_data['week_number'] = time_period_data['timestamp'].apply(lambda x: (x.day - 1) // 7 + 1)
        time_period_data['week_number'] = time_period_data['week_number'].apply(lambda x: min(x, 4))
        grouping_column = 'week_number'
        title = 'Error Types Distribution Over the Last 4 Weeks'
        xlabel = 'Weeks of the Month'
        ylabel = 'Count'
    elif time_period == '3M':
        # Format 'month_year' to show the name of the month and year
        time_period_data['month_year'] = time_period_data['timestamp'].dt.strftime('%B %Y')
        grouping_column = 'month_year'
        title = 'Error Types Distribution Over the Last 3 Months'
        xlabel = 'Month'
        ylabel = 'Count'

    # Create a pivot table and sort if it's for the '3M' time period
    error_counts = time_period_data.pivot_table(index=grouping_column, columns='type_of_error', values='error', aggfunc='sum', fill_value=0)

    if time_period == '3M':
        # Convert the index to datetime to sort it
        error_counts.index = pd.to_datetime(error_counts.index, format='%B %Y')
        error_counts.sort_index(inplace=True)
        # Convert back to month names for display
        error_counts.index = error_counts.index.strftime('%B %Y')

    # Create a vertical stacked bar chart for the error types
    plt.figure(figsize=(12, 6))
    sns.set_palette("Set2")  # Use a color palette
    ax = error_counts.plot(kind='bar', stacked=True)

    # Swapping the labels for the vertical orientation
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title='Error Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set x-tick labels according to the time period
    if time_period == '1W':
        ax.set_xticklabels(days_order, rotation=90)
    elif time_period == '3M':
        # The labels are already in the right format for '3M'
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    else:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    st.pyplot()

def plot_average_response_time(df, time_period):
    # Convert the 'timestamp' column to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Define the time frames for analysis
    latest_date = df['timestamp'].max().date()
    one_week_ago = latest_date - timedelta(weeks=1)
    one_month_ago = latest_date.replace(day=1) - timedelta(days=1)
    three_months_ago = latest_date.replace(day=1) - pd.DateOffset(months=2)

    # Filter data for each time frame
    week_data = df[(df['timestamp'].dt.date > one_week_ago) & (df['timestamp'].dt.date <= latest_date)]
    month_data = df[(df['timestamp'].dt.date > one_month_ago) & (df['timestamp'].dt.date <= latest_date)]
    three_months_data = df[(df['timestamp'] >= pd.Timestamp(three_months_ago)) & (df['timestamp'] <= pd.Timestamp(latest_date))]

    # Plot the relevant graph
    if time_period == '1 week':
        # Group data by day
        week_data_grouped = week_data.groupby(week_data['timestamp'].dt.day_name())['response_time'].mean().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

        # Plot for average response time during the latest week
        plt.figure(figsize=(10, 6))
        bars = plt.bar(week_data_grouped.index, week_data_grouped.values, color='blue')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center')
        plt.title('Average Response Time - Latest Week')
        plt.xlabel('Day of the Week')
        plt.ylabel('Average Response Time (seconds)')
    elif time_period == '1 month':
        # Calculate the start of the month
        start_of_month = month_data['timestamp'].min().replace(day=1)
        end_of_fourth_week = start_of_month + timedelta(weeks=4) - timedelta(days=1)
        month_data_filtered = month_data[(month_data['timestamp'] >= start_of_month) & 
                                        (month_data['timestamp'] <= end_of_fourth_week)]
        month_data_filtered['week_of_month'] = month_data_filtered['timestamp'].apply(
            lambda x: (x - start_of_month).days // 7 + 1)
        month_data_grouped = month_data_filtered.groupby('week_of_month')['response_time'].mean()
        plt.figure(figsize=(10, 6))
        bars = plt.bar(month_data_grouped.index, month_data_grouped.values, color='green')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center')
        plt.title('Average Response Time - First 4 Weeks of the Month')
        plt.xlabel('Week of the Month')
        plt.ylabel('Average Response Time (seconds)')
        plt.xticks(range(1, 5))
    elif time_period == '3 months':
        # Group data by month
        three_months_data['month_year'] = three_months_data['timestamp'].dt.strftime('%B %Y')
        three_months_data_grouped = three_months_data.groupby('month_year')['response_time'].mean()
        sorted_months = sorted(three_months_data_grouped.index, key=lambda x: pd.to_datetime(x))

        # Plot for average response time during the latest three months
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sorted_months, three_months_data_grouped[sorted_months].values, color='red')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center')
        plt.title('Average Response Time - Latest 3 Months')
        plt.xlabel('Month and Year')
        plt.ylabel('Average Response Time (seconds)')

    plt.xticks(rotation=45)
    st.pyplot()

def calculate_metrics_delta(df, latest_date, period):
    if period == "1 week":
        current_period_start = latest_date - pd.DateOffset(weeks=1)
        previous_period_start = latest_date - pd.DateOffset(weeks=2)
    elif period == "1 month":
        current_period_start = latest_date - pd.DateOffset(months=1)
        previous_period_start = latest_date - pd.DateOffset(months=2)
    elif period == "3 months":
        current_period_start = latest_date - pd.DateOffset(months=3)
        previous_period_start = latest_date - pd.DateOffset(months=6)

    current_users_count = df[(df['timestamp'] >= current_period_start) & (df['timestamp'] <= latest_date)]['user_id'].nunique()
    previous_users_count = df[(df['timestamp'] >= previous_period_start) & (df['timestamp'] < current_period_start)]['user_id'].nunique()

    current_queries_count = df[(df['timestamp'] >= current_period_start) & (df['timestamp'] <= latest_date)].shape[0]
    previous_queries_count = df[(df['timestamp'] >= previous_period_start) & (df['timestamp'] < current_period_start)].shape[0]

    delta_users = current_users_count - previous_users_count
    delta_queries = current_queries_count - previous_queries_count

    return current_users_count, delta_users, current_queries_count, delta_queries

# UI Layout
def main_layout():
    with st.sidebar:
        st.image(logo, width=300)
        st.title("Menu")
        st.markdown("---")

        if st.button("📊 Dashboard"):
            st.session_state['current_tab'] = 'Dashboard'
        if st.button("💬 Conversation"):
            st.session_state['current_tab'] = 'Conversation'

        st.markdown("---")
        st.markdown("## About")
        st.info("Devan & Company's data scientists and analysts are trained experts in analyzing, cleaning and transforming your data to create models that highlight the most relevant information pertaining to your business.")

    if st.session_state['current_tab'] == 'Conversation':
        conversation_tab()
    elif st.session_state['current_tab'] == 'Dashboard':
        dashboard_tab()

def conversation_tab():
    st.subheader("Chat")
    
    # Define the columns you want to display
    columns_to_display = ['timestamp', 'user_id', 'user_message', 'bot_response']
    
    # Create a subset of the DataFrame with the selected columns
    subset_df = df[columns_to_display]
    
    # Display the subset of the DataFrame
    st.write(subset_df.sort_values(by='timestamp', ascending=False))

def dashboard_tab():
    col1a, col2a, col3a, col4a, col5a = st.columns(5)
    with col1a:
        st.subheader("Dashboard")
    with col2a, col3a, col4a, col5a:
        st.empty()
    with col5a:
        time_delta_option = st.selectbox("Select Time Period", ["1 week", "1 month", "3 months"])

    col1b, col2b, col3b = st.columns(3)
    # Latest date in the dataset
    latest_date = df['timestamp'].max()
    with col1b:
        st.header("Total Users")
        users_count, delta_users, _, _ = calculate_metrics_delta(df, latest_date, time_delta_option)
        st.metric(label=f"Last {time_delta_option}", value=users_count, delta=f"{delta_users}")

    with col2b:
        st.header("Total Queries")
        _, _, queries_count, delta_queries = calculate_metrics_delta(df, latest_date, time_delta_option)
        st.metric(label=f"Total Queries - Last {time_delta_option}", value=queries_count, delta=f"{delta_queries}")
    
    with col3b:
        if time_delta_option == "1 week":
            sentiment_analysis(df, 'week', 'For Latest Week')
        elif time_delta_option == "1 month":
            sentiment_analysis(df, 'month', 'For Latest Month')
        elif time_delta_option == "3 months":
            sentiment_analysis(df, '3month', 'For Latest 3 Months')
            
    col1c, col2c = st.columns(2)
    with col1c:
        if time_delta_option == "1 week":
            users_queries_line_graph(df, 'week')
        elif time_delta_option == "1 month":
            users_queries_line_graph(df, 'month')
        elif time_delta_option == "3 months":
            users_queries_line_graph(df, '3months')
    with col2c:
        if time_delta_option == "1 week":
            plot_error_types_distribution(df, time_period='1W')

        elif time_delta_option == "1 month":
            plot_error_types_distribution(df, time_period='1M')
        
        elif time_delta_option == "3 months":
            plot_error_types_distribution(df, time_period='3M')
    
    col1d, col2d = st.columns(2)
    with col1d:
        if time_delta_option == "1 week":
            plot_average_response_time(df, '1 week')
        elif time_delta_option == "1 month":
            plot_average_response_time(df, '1 month')
        elif time_delta_option == "3 months":
            plot_average_response_time(df, '3 months')

    with col2d:
        # Display heatmap based on the selected time period
        if time_delta_option == "1 week":
            create_heatmap(df, 7, "User Activity Heatmap for Last 7 Days")
        elif time_delta_option == "1 month":
            create_heatmap(df, 28, "User Activity Heatmap for Last Month")
        elif time_delta_option == "3 months":
            create_heatmap(df, 84, "User Activity Heatmap for Last 3 Months")

# Initialize session state
if 'current_tab' not in st.session_state:
    st.session_state['current_tab'] = 'Dashboard'

# Load your company logo
logo = Image.open('./source/devan&company.png')

# Main
main_layout()