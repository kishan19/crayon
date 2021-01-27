import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import joblib
import scipy

st.title('Recommendation Engine Task 1')

st.write("Visualising the results of the recommendation engine")


reco_model=joblib.load('reco_model.joblib')
all_users=pd.read_csv('./data/all_users.csv')
user_scaled=pd.read_csv('./data/user_scaled.csv')
cluster_results_for_profile_reco=pd.read_csv('./data/cluster_results_for_profile_reco.csv')
cluster_persons=pd.read_csv('./data/cluster_persons.csv')

def users_prediction(user_id):
    '''Return ranked offers for user by model'''
    test_user=all_users.loc[all_users['user_id']==user_id]
    test_user_test=test_user.drop(columns=['user_id','offer_id','rating'])
    recommender_offer_scores=list(reco_model.predict(test_user_test))
    recommender_offers=test_user['offer_id'].values.tolist()
    
    offers_ranked=[x for _,x in sorted(zip(recommender_offer_scores,recommender_offers))]
    
    return offers_ranked

from scipy import spatial

def compute_cosine_similarity(user1_id, user2_id,user_df):
    
    #get user rows
    user1 = user_df.loc[user1_id]
    user2 = user_df.loc[user2_id]
    
    cos_dist = 1 - spatial.distance.cosine(user1, user2)
    
    return cos_dist


def find_most_similar_users(user,user_df,k=5):
    '''Return k most similar users to a particular user iD'''
    sim_list=[]
    for idx in user_df.index:
        sim = compute_cosine_similarity(user, idx,user_df)
        sim_list.append(sim)
    #convert to numpy array
    sim_list= np.array(sim_list)
    #Sort in desceding order of similarity
    most_sim_users = sim_list.argsort()[::-1]
    #drop the first user since it is same user
    most_sim_users = most_sim_users[1:][:k]
        
    return list(most_sim_users)


##Interactive layout for surfacing recommendations

if st.checkbox('New user'):
    if st.checkbox("Profile of user known"):
        st.write("Awesome..enter profile details as below..")
        selected_gender=st.selectbox("Select gender",["M","F","O"])
        selected_income=st.number_input("write income in 1000s",min_value=30000)
        selected_age=st.number_input("Enter age",min_value=18,max_value=100)
        
        st.write("Producing recommendations based on similar profiles....")

        profile_test=pd.DataFrame({'Income':[selected_income],'Age':[selected_age],'F':[0],'M':[0],'O':[0]})
        profile_test.loc[0,selected_gender]=1  

        ary = scipy.spatial.distance.cdist(profile_test, cluster_results_for_profile_reco,metric='euclidean')

        cluster_most_similar=np.argmax(ary==ary.min())+1

        st.write("Based on the profile entered, this is most close to Cluster # {0}".format(cluster_most_similar))

        cluster_person_chosen=np.random.choice(cluster_persons.loc[cluster_persons['cluster']==cluster_most_similar].person,1)[0]
        st.write("Hence you may provide recommendations which you do for person # {0}".format(int(cluster_person_chosen)))



    else:
        st.write("If profile unknown, we will produce default recommendations..This is the most offers viewed by people")
        offer_default_df=pd.DataFrame({'Most viewed offers by whole set of viewers':[10,2,9,5,8]})
        offer_default_df


else:
    ##If it is not a new user
    option=st.sidebar.selectbox("Select user ID",range(0,17000))
    st.write("Producing recommendations for user {0}".format(option))

    if option==0:
        st.write("Select a user ID from 1 to 17000")

    else:

        user_id=int(option)
        non_cold_start_recos=[]

        model_recommendations_by_interaction=users_prediction(user_id)
        ##If this is empty, produce category

        if len(model_recommendations_by_interaction)==0:
            st.write("Cold start user within the data...producing recommendations based on 10 Nearest neighbours..")
            user_category='Cold start'

        else:
            user_category='Normal'
            for mr in model_recommendations_by_interaction:
                non_cold_start_recos.append(mr)
            
        ##Show nearby predictions

        if user_category=='Cold start':
            similar_users=find_most_similar_users(user_id,user_scaled,10)

        else:
            similar_users=find_most_similar_users(user_id,user_scaled,5)

        offers_similar=[]
        for su in similar_users:
            model_recommendations_for_neighbour=users_prediction(su)
            if len(model_recommendations_for_neighbour)==0:
                pass
            else:
                for mrn in model_recommendations_for_neighbour:
                    if mrn not in non_cold_start_recos:
                        non_cold_start_recos.append(mrn)

        result_df=pd.DataFrame({'Recommendations for user':non_cold_start_recos})

        result_df

  