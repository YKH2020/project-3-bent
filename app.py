import streamlit as st
from time import sleep
from naive import recommend_top_joke
from get_jokes import read_clean_joke_text
from input_joke import get_best_joke_for_user

users = {
    "Bill": 2,
    "Tejas": 3,
    "Aryan": 4,
    "Yash": 5
}

def run_naive_approach(user):
    with st.spinner(text="Processing", show_time=True):
        joke_id = recommend_top_joke("user_" + str(users.get(user)))
        st.write(read_clean_joke_text(joke_id)[joke_id])

def run_non_dl(user):
    with st.spinner(text="Processing", show_time=True):
        joke_id = get_best_joke_for_user("user_" + str(users.get(user)))
        st.write(read_clean_joke_text(joke_id)[joke_id])

def run_dl():
    with st.spinner(text="Processing", show_time=True):
        sleep(5)


st.title("Joke Recommender")
st.write("This tool recommends jokes for users based on their ratings on a set of 10 jokes")

st.header("Getting started")
st.write("To get started on using the models below, start by selecting which user's recommendations you want to view.")

user = st.selectbox(
    "Select User",
    ("Bill", "Tejas", "Aryan", "Yash")
)

st.header("The Naive Approach")
st.write("For the naive approach we built a mean model in order to recommend the top joke for " + f'{user}')

# if st.button("Run Naive Approach"):
    # run_naive_approach(user)

st.header("Traditional ML Method")
st.write("In the non-deep learning method we used probabilistic matrix factorization to recommend the top joke for " + f'{user}')

if st.button("Run PMF"):
    run_non_dl(user)

st.header("Deep Learning Method")
st.write("This model was a custom autoencoder used to train on our dataset to recommend the top joke for " + f'{user}')

if st.button("Run Autoencoder"):
    run_dl()