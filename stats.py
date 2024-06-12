from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji

extractor = URLExtract()

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # 1. Number of messages
    no_of_messages = df.shape[0]

    # 2. Total number of words in all messages
    words = []
    for message in df['message']:
        words.extend(message.split())
    no_of_words = len(words)

    # Number of media messages
    no_of_media = df[df['message'] == '<Media omitted>\n'].shape[0]

    #Number of links
    links = []
    for message in df['message']:
        links.extend(extractor.find_urls(message))
    return no_of_messages,no_of_words,no_of_media,len(links)

def most_busy_users(df):
    X = df['user'].value_counts().head()
    percent_df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'user': 'name', 'count': 'percent'})
    return X, percent_df

def monthly_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + " " + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline

def daily_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    daily_timeline = df.groupby('day_date').count()['message'].reset_index()
    return daily_timeline

def weekly_activity(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def monthly_activity(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()

def activity_heatmap(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    heatmap_table = df.pivot_table(index='day_name', columns='time_period', values='message', aggfunc='count').fillna(0)
    return heatmap_table

def create_wordcloud(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    temp_df = df[df['user'] != 'group notification']
    temp_df = temp_df[temp_df['message'] != '<Media omitted>\n']
    def remove_stopwords(message):
        no_stopwords = []
        for word in message.lower().split():
            if word not in stop_words:
                no_stopwords.append(word)
        return " ".join(no_stopwords)

    temp_df['message'] = temp_df['message'].apply(remove_stopwords)

    wc = WordCloud(width = 400, height = 300, min_font_size = 10, background_color = 'white')
    df_wc = wc.generate(temp_df['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    temp_df = df[df['user'] != 'group notification']
    temp_df = temp_df[temp_df['message'] != '<Media omitted>\n']
    words = []
    for message in temp_df['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    common_words_df =  pd.DataFrame(Counter(words).most_common(20))
    return common_words_df

#emoji analysis
def emoji_info(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = []
    for message in df['message']:
        emojis.extend([ch for ch in message if emoji.is_emoji(ch)])
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(emojis)))
    return emoji_df