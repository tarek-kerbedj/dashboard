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

# Enable wide mode
st.set_page_config(layout="wide")

# Load data from CSV
df = pd.read_csv('./data/chat_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Modify the create_heatmap function
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

def sentiment_analysis(df, period, period_title):
    """
    Performs sentiment analysis based on the period provided and plots a radar chart.
    :param df: DataFrame with the data
    :param period: A string specifying the period ('week', 'month', or '3month')
    :param period_title: A descriptive title for the period
    """
    # Get the most recent date in your dataset
    end_date = df['timestamp'].max()
    start_date = end_date - pd.Timedelta(days=6)  # Calculate the start date for the last 7 days

    # Filter the DataFrame for the last 7 days
    recent_week_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

    # Adjust the pivot table generation based on the period
    if period == 'week':
        # Aggregate data over the last 7 days
        pivot_data = recent_week_df.pivot_table(columns='sentiment', values='timestamp', aggfunc='count').sum().to_frame().T
    elif period == 'month':
        # Here, instead of using day, use week of the month for grouping
        df['week_of_month'] = df['timestamp'].dt.isocalendar().week - df['timestamp'].dt.isocalendar().week.min() + 1
        pivot_data = df.pivot_table(index='week_of_month', columns='sentiment', values='timestamp', aggfunc='count').head(4)  # Limit to the first 4 weeks

    elif period == '3month':
        df['month_year'] = df['timestamp'].dt.to_period('M')
        pivot_data = df.pivot_table(index='month_year', columns='sentiment', values='timestamp', aggfunc='count').iloc[-3:]

    # Convert pivot_data to a format suitable for radar chart
    categories = list(pivot_data.columns)
    N = len(categories)

    # Calculate the angles for the radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Radar chart setup
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Draw one axe per sentiment
    for idx, row in pivot_data.iterrows():
        values = row.tolist()
        values += values[:1]  # Repeat the first value to close the circle
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=idx)
        ax.fill(angles, values, alpha=0.2)

    # Add labels with emojis for each sentiment
    emoji_dict = {
        "Positive": "😃",
        "Neutral": "😐",
        "Negative": "😢",
    }

    labels = [emoji_dict.get(cat, cat) for cat in categories]  # Replace category with emoji if available

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)  # Use labels with emojis

    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(f'Sentiment Analysis ({period_title})')

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

def line_graph(df, period):
    end_date = df['timestamp'].max()
    start_date = get_start_date(end_date, period)

    if period == 'week':
        period_data = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        counts = period_data.groupby(period_data['timestamp'].dt.date).size()
        counts.index = pd.to_datetime(counts.index)
        x_labels = counts.index.strftime('%A')
        
        # Calculate the number of users per day
        users_per_day = period_data.groupby(period_data['timestamp'].dt.date)['user_id'].nunique()

    elif period == 'month':
        period_data = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        period_data['week_of_month'] = period_data['timestamp'].apply(lambda x: (x.day - 1) // 7 + 1)
        counts = period_data.groupby('week_of_month').size()
        x_labels = range(1, 6)
        
        # Calculate the number of users per week
        users_per_week = period_data.groupby('week_of_month')['user_id'].nunique()

    elif period == '3months':
        period_data = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        counts = period_data.groupby(period_data['timestamp'].dt.to_period('M')).size()
        counts.index = counts.index.to_timestamp()
        x_labels = counts.index.strftime('%Y-%m')
        
        # Calculate the number of users per month
        users_per_month = period_data.groupby(period_data['timestamp'].dt.to_period('M'))['user_id'].nunique()

    plt.figure(figsize=(10, 6))
    plt.plot(counts.index, counts.values, marker='o', label='Number of Queries', linestyle='-', color='blue')
    plt.title(f'Queries and Users in the Latest {period.capitalize()}')
    plt.xlabel('Time Period')
    plt.ylabel('Count')
    plt.xticks(ticks=counts.index, labels=x_labels, rotation=45)
    
    # Plot the number of users on the same graph
    if period == 'week':
        plt.plot(users_per_day.index, users_per_day.values, marker='s', linestyle='--', color='red', label='Number of Users')
    elif period == 'month':
        plt.plot(users_per_week.index, users_per_week.values, marker='s', linestyle='--', color='red', label='Number of Users')
    elif period == '3months':
        plt.plot(users_per_month.index, users_per_month.values, marker='s', linestyle='--', color='red', label='Number of Users')
    
    plt.legend(loc='upper left', bbox_to_anchor=(0.7, 1.0))  # Adjust legend position
    plt.grid(True)
    st.pyplot()

def plot_error_types_distribution(df, time_period):
    # Set time period start and end dates
    end_date = df['timestamp'].max()
    if time_period == '1W':
        start_date = end_date - pd.DateOffset(days=6)
    elif time_period == '1M':
        start_date = pd.Timestamp(end_date) - pd.DateOffset(weeks=4)  # Convert to Timestamp and adjust to 4 weeks
    elif time_period == '3M':
        start_date = pd.Timestamp(end_date) - pd.DateOffset(months=3)  # Convert to Timestamp and adjust to 3 complete months

    # Convert start_date to datetime
    start_date = pd.Timestamp(start_date)

    # Filter data for the specified time period
    time_period_data = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]


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

    if time_period == '1W':
        ax.set_xticks(range(len(days_order)))
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
    # Calculate the response time
    df['response_time'] = df['timestamp'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()

    # Define the time frames for analysis
    latest_date = df['timestamp'].max().date()
    one_week_ago = latest_date - timedelta(weeks=1)
    one_month_ago = latest_date.replace(day=1) - timedelta(days=1)  # The start of the current month
    three_months_ago = latest_date.replace(day=1) - pd.DateOffset(months=2)  # The start of the month three months ago

    # Filter data for each time frame
    week_data = df[(df['timestamp'].dt.date > one_week_ago) & (df['timestamp'].dt.date <= latest_date)]
    month_data = df[(df['timestamp'].dt.date > one_month_ago) & (df['timestamp'].dt.date <= latest_date)]
    three_months_data = df[(df['timestamp'] >= pd.Timestamp(three_months_ago)) & (df['timestamp'] <= pd.Timestamp(latest_date))]

    # Based on the time period, plot the relevant graph
    if time_period == '1 week':
        plot_weekly_data(week_data)
    elif time_period == '1 month':
        plot_monthly_data(month_data)
    elif time_period == '3 months':
        plot_three_months_data(three_months_data)

def plot_weekly_data(week_data):
    # Group data by day
    week_data_grouped = week_data.groupby(week_data['timestamp'].dt.day_name())['response_time'].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    # Plot for average response time during the latest week
    plt.figure(figsize=(10, 6))
    plt.plot(week_data_grouped.index, week_data_grouped.values, marker='o', linestyle='-', color='blue')
    plt.title('Average Response Time - Latest Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Average Response Time (seconds)')
    plt.xticks(rotation=45)
    st.pyplot()

def plot_monthly_data(month_data):
    # Group data by week
    month_data['week_of_month'] = month_data['timestamp'].dt.isocalendar().week
    month_data_grouped = month_data.groupby('week_of_month')['response_time'].mean()

    # Plot for average response time during each week of the month
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(month_data_grouped) + 1), month_data_grouped.values, marker='o', linestyle='-', color='green')
    plt.title('Average Response Time - Latest Month')
    plt.xlabel('Week of the Month')
    plt.ylabel('Average Response Time (seconds)')
    st.pyplot()

def plot_three_months_data(three_months_data):
    # Group data by month
    three_months_data['month_year'] = three_months_data['timestamp'].dt.strftime('%B %Y')
    three_months_data_grouped = three_months_data.groupby('month_year')['response_time'].mean()

    # Sort the months in chronological order
    sorted_months = sorted(three_months_data_grouped.index, key=lambda x: pd.to_datetime(x))

    # Plot for average response time during the latest three months
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_months, three_months_data_grouped[sorted_months].values, marker='o', linestyle='-', color='red')
    plt.title('Average Response Time - Latest 3 Months')
    plt.xlabel('Month and Year')
    plt.ylabel('Average Response Time (seconds)')
    plt.xticks(rotation=45)
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
        st.info("Devan & Company's data scientists and analysts are trained experts in analyzing, cleaning and transforming your data to create models that highlight the most relevant information pertaining to your business.")

    if st.session_state['current_tab'] == 'Conversation':
        conversation_tab()
    elif st.session_state['current_tab'] == 'Dashboard':
        dashboard_tab()

def conversation_tab():
    st.subheader("Chat")
    st.write(df.sort_values(by='timestamp', ascending=False))

def dashboard_tab():
    col1a, col2a, col3a, col4a, col5a, col6a = st.columns(6)
    with col1a:
        st.subheader("Dashboard")
    with col2a, col3a, col4a, col5a:
        st.empty()
    with col6a:
        time_delta_option = st.selectbox("Select Time Period", ["1 week", "1 month", "3 months"])
        time_delta = {"1 week": 7, "1 month": 30, "3 months": 90}[time_delta_option]

    col1b, col2b = st.columns(2)
    with col1b:
        create_heatmap(df, time_delta)
    with col2b:
        if time_delta_option == "1 week":
            sentiment_analysis(df, 'week', 'Day of the Week')
        elif time_delta_option == "1 month":
            sentiment_analysis(df, 'month', 'Week of the Month')
        elif time_delta_option == "3 months":
            sentiment_analysis(df, '3month', 'Month')
    col1c, col2c = st.columns(2)
    with col1c:
        if time_delta_option == "1 week":
            line_graph(df, 'week')
        elif time_delta_option == "1 month":
            line_graph(df, 'month')
        elif time_delta_option == "3 months":
            line_graph(df, '3months')
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
        st.empty()

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