from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval
from deepeval import evaluate

from Chatbot import graph
from langchain_core.messages import AIMessage, HumanMessage

# !!!!!
# Make sure you run this file and any others from WITHIN THEIR DIRECTORY. 
# Running this file from outside its directory will cause an IoError. 
# !!!!!

def testChatbot(question):
    # I'm not aiming to test the conversational memory, but rather it's retrieval ability uninfluenced by existing conversation.
    # Therefore, every time the chatbot is invoked, it'll only know this message, which is the same as when it is initially opened in Streamlit.
    evalConversation = [AIMessage(content="Hello! I'm an assistant chatbot designed to help answer any questions you have about BCU."),
                        HumanMessage(content = question)]

    response = graph.invoke({
        "messages": evalConversation 
    })
    
    # Returns the most recent message, which is the chatbot's response.
    return response["messages"][-1].content


# Nested list of the question/answer pairs to be converted to DeepEval's LLMTestCases.
qaPairs = [
    ["What happens if I submit my assignment 3 days late?", "If an assignment is submitted 3 days late, your mark will be reduced by 10%."],
    ["What happens if I submit my assignment 3 minutes late?", "If an assignment is submitted 3 minutes late, it is not considered a late submission, and your mark will not be reduced."],
    ["What is an EC claim?", "An Extenuating Circumstances claim can be made by a student if there are circumstances that affect their ability to submit assessments on time, complete assessments to a good standard or attend in-person assessments."],
    ["What circumstances will be accepted as extenuating circumstances?", "Serious short-term illness or injury, worsening of an ongoing illness or disability, symptoms of a harmful infectious disease, death or significant illness of a close family member or friend, unexpected caring responsibilities, significant personal or family crises, witnessing or experiencing a traumatic incident, a crime which has had a substantial impact on you, an accommodation crisis such as eviction."],
    ["When must I enrol?", "Students must enrol at the start of their programme and enrol for each level by the Friday of week four from the start date of their course unless a Break in Study has been approved."],
    ["What is the pass mark for a module?", "For an undergraduate course, the pass mark is 40%. On a postgraduate course, it is instead 50%."],
    ["What happens if I fail a module?", "The first time you fail a module, you can be re-assessed for failed assessments, which is known as a resit. Your grade will be capped at the pass mark. You cannot be reassessed for assessments that you passed."],
    ["What are the degree classifications?", "Achieving an average of 70% or above grants you first-class honours. 60-69% is an upper second (2:1), 50-59% is a lower second (2:2), 40-49% is third-class honours, and anything below 40% is a fail."],
    ["How many BCU students are there?", "BCU has 31,300 students as of 2022-23."],
    ["What is BCUSU?", "The Birmingham City University Student Union represent you as a student, and work together with the university to make change. They can help with various academic topics and offer societies to help people make friends."]
]

testCases = []

# Creates DeepEval's LLMTestCases for each question and answer pair for input and expected output.
# For the actual output, the chatbot is used.
for qa in qaPairs:
    testCases.append(
        LLMTestCase(input = qa[0],
                    actual_output = testChatbot(qa[0]),
                    expected_output = qa[1])
    )
    
# for test in testCases:
#     print("\n")
#     print(test.input)
#     print(test.actual_output)
#     print(test.expected_output)

# Create an evaluation dataset composed of the test cases.
dataset = EvaluationDataset(test_cases = testCases)

correctness_metric = GEval(
    name="Correctness",
    criteria = "Determine whether the actual output is factually correct based on the expected output. Any additional scenarios or details not present in the expected output are OK.",
    model="gpt-4o",
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ]
)

# Evaluate the dataset using the defined GEval metric.
evaluation_results = evaluate(dataset, metrics = [correctness_metric])

