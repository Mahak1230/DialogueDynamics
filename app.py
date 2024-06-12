import matplotlib.pyplot as plt
import preprocessor,stats
import streamlit as st
import matplotlib
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
st.sidebar.title('DialogueDynamics')

uploaded_file = st.sidebar.file_uploader('Choose a file')
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode('utf-8')
    df = preprocessor.preprocess(data)
    #printing the dataframe
    # st.dataframe(df)
    user_list = df['user'].unique().tolist()
    if 'group notification' in user_list:
        user_list.remove('group notification')
    user_list.insert(0,'Overall')

    selected_user = st.sidebar.selectbox('Analysis according to group or dm', user_list)

    if st.sidebar.button('Show Analysis'):
        no_of_messages,no_of_words,no_of_media,no_of_links = stats.fetch_stats(selected_user,df)
        st.title("Top Statistics")
        col1, col2, col3,col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(no_of_messages)
        with col2:
            st.header("Total Words")
            st.title(no_of_words)
        with col3:
            st.header("Media Shared")
            st.title(no_of_media)
        with col4:
            st.header("Links Shared")
            st.title(no_of_links)

        #Monthly Timeline
        st.title("Monthly Timeline")
        timeline = stats.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'],timeline['message'],color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        #Daily Timeline
        st.title("Daily Timeline")
        daily_timeline = stats.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['day_date'], daily_timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        #Activity map
        st.title('Weekly Acitivity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_days = stats.weekly_activity(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_days.index, busy_days.values,color='cyan')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = stats.monthly_activity(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='pink')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # Heatmap
        st.title("Weekly Activity Heatmap")
        heatmap_table = stats.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(heatmap_table)
        plt.yticks(rotation='horizontal')
        st.pyplot(fig)

#in doubt, should display before or after the triggering of the 'show analysis' button
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            X, percent_df = stats.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(X.index, X.values,color = [0.8,0.1,0.1,0.8])
                plt.xticks(rotation = 'vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(percent_df)

            st.title("Wordcloud")
            df_wc = stats.create_wordcloud(selected_user,df)
            fig,ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)

            most_common_df =   stats.most_common_words(selected_user,df)
            fig, ax = plt.subplots()
            ax.bar(most_common_df[0], most_common_df[1],color = 'purple')
            plt.xticks(rotation='vertical')
            st.title('Most Common Words')
            st.pyplot(fig)

            emoji_df = stats.emoji_info(selected_user,df)
            st.title("Emoji Analysis")
            if emoji_df.empty:
                st.write("No emojis are used")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(emoji_df)
                with col2:
                    fig, ax = plt.subplots()
                    ax.pie(emoji_df[1], labels=emoji_df[0], autopct='%0.2f')
                    # ax.bar(emoji_df[0], emoji_df[1],color='blue')
                    st.pyplot(fig)


            # Load the trained model

            # Button to trigger toxicity analysis
            st.title('Toxicity Analysis')

            if st.button('Analyze'):
                model = load_model('toxicity.h5')
                # model.layers[0].input_length = 10
                # Preprocess input message
                vectorizer = TextVectorization(max_tokens=10000,
                                               output_sequence_length=1000,
                                               output_mode='int')
                df_t = pd.read_csv('train.csv')
                X = df_t['comment_text']
                vectorizer.adapt(X.values)
                toxicity_class = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
                toxic_messages = {cls: [] for cls in toxicity_class}
                no_toxic_message = True

                for index, message in enumerate(df['message']):
                    vectorized_message = vectorizer([message])  # Transform message into numerical vector
                    prediction = model.predict(vectorized_message)  # Predict probabilities for each class
                    predicted_classes = [toxicity_class[i] for i, prob in enumerate(prediction[0]) if prob > 0.5]
                    if predicted_classes:
                        no_toxic_message = False
                        for cls in predicted_classes:
                            toxic_messages[cls].append(message)

                if no_toxic_message:
                    st.write("There are no toxic messages.")
                else:
                    for cls, messages in toxic_messages.items():
                        if messages:
                            st.write(f"The message(s) which are '{cls}' are the following:")
                            for message in messages:
                                st.markdown(f"- {message}")
