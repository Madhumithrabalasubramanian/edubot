import pandas as pd
import streamlit as st
from streamlit_chat import message
from ctransformers import AutoModelForCausalLM

# Load the dataset
data = pd.read_excel('data/dataset.xlsx')

# Load the LLaMA model using CTransformers
llm_pipeline = AutoModelForCausalLM.from_pretrained("TheBloke/LLaMA-2-7B-GGML", model_type="llama")

# Function to fetch college details
def get_college_info(college_name):
    college_info = data[data['College Name'].str.contains(college_name, case=False)]
    if college_info.empty:
        return None
    return college_info.iloc[0]  # Return the first matching college

# Function to list colleges in a specific location
def list_colleges_in_location(location):
    colleges = data[data['Location'].str.contains(location, case=False)]
    if colleges.empty:
        return "No colleges found in that location."
    return colleges['College Name'].tolist()

# Function to describe college in detail
def describe_college(college_info):
    college_name = college_info['College Name']
    location = college_info['Location']
    ownership = college_info['Ownership']
    accreditation = college_info['Accreditation']
    institution_type = college_info['Type of Institution']
    admission_criteria = college_info['Admission Criteria']
    application_process = college_info['Application Process']
    application_fees = college_info['Application Fees']
    scholarships = college_info['Scholarship Opportunities']
    entrance_exams = college_info['Entrance Exams']
    programs_offered = college_info['Programs Offered']
    duration_of_programs = college_info['Duration of Programs']
    curriculum_highlights = college_info['Curriculum Highlights']
    tuition_fees = college_info['Tuition Fees']
    other_fees = college_info['Other Fees']
    payment_plans = college_info['Payment Plans']
    campus_facilities = college_info['Campus Facilities']
    placement_statistics = college_info['Placement Statistics']
    internship_opportunities = college_info['Internship Opportunities']
    email = college_info['Email Address']
    website = college_info['Website URL']

    return f"""
    {college_name}, located in {location}, operates under {ownership} and holds {accreditation}. 
    As a {institution_type}, it offers {programs_offered} across various fields. 
    The admission criteria require applicants to follow the application process, which includes an application fee of ${application_fees}. 
    Scholarship opportunities are available to assist financially. 
    Entrance exams may be required for admission. 
    The programs offered span a duration of {duration_of_programs} and cover {curriculum_highlights}. 
    Tuition fees are ${tuition_fees} and other fees are ${other_fees}, with payment plans available. 
    The campus facilities support various activities. 
    Placement statistics are impressive, and internship opportunities provide industry experience. 
    For more information, contact {email} or visit {website}.
    """

# Function to handle user queries and provide specific details
def handle_query(college_info, query):
    responses = {
        "location": f"The location of the specific college is in {college_info['Location']}.",
        "ownership": f"The college is under {college_info['Ownership']} ownership.",
        "accreditation": f"The college is accredited by {college_info['Accreditation']}.",
        "type of institution": f"It is a {college_info['Type of Institution']}.",
        "admission criteria": f"The admission criteria include {college_info['Admission Criteria']}.",
        "application process": f"The application process involves {college_info['Application Process']}.",
        "application fees": f"The application fee is ${college_info['Application Fees']}.",
        "scholarship opportunities": f"The available scholarship opportunities include {college_info['Scholarship Opportunities']}.",
        "entrance exams": f"The required entrance exams are {college_info['Entrance Exams']}.",
        "programs offered": f"The programs offered are {college_info['Programs Offered']}.",
        "duration of programs": f"The duration of the programs is {college_info['Duration of Programs']}.",
        "curriculum highlights": f"The curriculum highlights include {college_info['Curriculum Highlights']}.",
        "tuition fees": f"The tuition fees amount to ${college_info['Tuition Fees']}.",
        "other fees": f"Other fees include ${college_info['Other Fees']}.",
        "payment plans": f"The payment plans available are {college_info['Payment Plans']}.",
        "campus facilities": f"The campus facilities include {college_info['Campus Facilities']}.",
        "placement statistics": f"The placement statistics show {college_info['Placement Statistics']}.",
        "internship opportunities": f"The internship opportunities available are {college_info['Internship Opportunities']}.",
        "email": f"For inquiries, contact {college_info['Email Address']}.",
        "website": f"More information is available at {college_info['Website URL']}.",
    }

    for keyword, response in responses.items():
        if keyword in query.lower():
            return response

    return "I'm sorry, I didn't understand that. Please ask about a specific aspect of the college."

# Function to compare two colleges
def compare_colleges(college_a, college_b):
    info_a = get_college_info(college_a)
    info_b = get_college_info(college_b)

    if info_a is None or info_b is None:
        return "One or both college names are incorrect."

    comparison = f"""
    **Comparison between {college_a} and {college_b}**:

    **{college_a}**:
    - Tuition Fees: ${info_a['Tuition Fees']}
    - Placement Statistics: {info_a['Placement Statistics']}

    **{college_b}**:
    - Tuition Fees: ${info_b['Tuition Fees']}
    - Placement Statistics: {info_b['Placement Statistics']}

    Based on the comparison, {college_a if info_a['Tuition Fees'] < info_b['Tuition Fees'] else college_b} is the better choice due to lower fees.
    """
    return comparison

# Initialize session state
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'current_college' not in st.session_state:
        st.session_state['current_college'] = None
    if 'list_location' not in st.session_state:
        st.session_state['list_location'] = False
    if 'compare_colleges' not in st.session_state:
        st.session_state['compare_colleges'] = False

# Display chat history
def display_chat_history():
    reply_container = st.container()
    container = st.container()
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Ask about a College:", placeholder="Type your question here...", key='input')
            submit_button = st.form_submit_button(label='Send')
        if submit_button and user_input:
            st.session_state['past'].append(user_input)
            handle_user_input(user_input)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state['generated'][i], key=str(i), avatar_style="fun-emoji")

# Function to handle user input
def handle_user_input(user_input):
    # Check for greetings
    if any(greet in user_input.lower() for greet in ['hello', 'hi', 'hii']):
        st.session_state['generated'].append("Hello! I'm happy to assist you with any college information you need. Please mention the college name.")
        return

    # If the user has already specified a college, continue with that college
    if st.session_state['current_college'] is not None:
        if "change college" in user_input.lower():
            st.session_state['current_college'] = None
            st.session_state['generated'].append("Please specify the college name you want to know about.")
            return

        # Check if the user is asking for specific information
        college_info = get_college_info(st.session_state['current_college'])
        if college_info is not None:
            answer = handle_query(college_info, user_input)
            st.session_state['generated'].append(answer)
            return

    # If no current college is set, look for a college name in the input
    if st.session_state['current_college'] is None:
        college_info = get_college_info(user_input)
        if college_info is not None:
            st.session_state['current_college'] = college_info['College Name']
            st.session_state['generated'].append(describe_college(college_info))
            return

    # Check for listing colleges
    if "list colleges" in user_input.lower():
        st.session_state['generated'].append("Please specify the location to list colleges.")
        st.session_state['list_location'] = True
        return

    # Check for location input for listing
    if st.session_state.get('list_location'):
        location = user_input
        colleges = list_colleges_in_location(location)
        st.session_state['generated'].append(f"Colleges in {location}: {', '.join(colleges) if isinstance(colleges, list) else colleges}.")
        st.session_state['list_location'] = False
        return

    # Check for compare request
    if "compare" in user_input.lower():
        st.session_state['generated'].append("Please provide the names of the colleges you want to compare, separated by 'and'.")
        st.session_state['compare_colleges'] = True
        return

    # Handle college comparison
    if st.session_state.get('compare_colleges'):
        college_names = [name.strip() for name in user_input.split("and")]
        if len(college_names) == 2:
            comparison_result = compare_colleges(college_names[0], college_names[1])
            st.session_state['generated'].append(comparison_result)
        else:
            st.session_state['generated'].append("Please specify exactly two colleges to compare.")
        st.session_state['compare_colleges'] = False
        return

    # Default response
    st.session_state['generated'].append("I'm sorry, I didn't understand that. Please ask again.")

# Initialize session state
initialize_session_state()

# Layout the application
st.title("🎓 College Infobot 📚")
display_chat_history()