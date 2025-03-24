from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval
from deepeval import evaluate

# Define test cases with input, actual output, and expected output.
# The 'actual_output' of each test case was copied and pasted directly from the chatbot's response.
test1 = LLMTestCase(
    input="What happens if I submit my assignment 3 days late?",
    actual_output="If you submit your assignment 3 days (between 24 hours and one week) late, your original mark will be reduced by 10%. For example, if your original mark was 60%, it would be reduced to 54%. However, this reduction will not apply if your original mark is below 40% (or 50% for postgraduate courses) or if the deduction would lower your mark from a pass to a fail. Remember, work will not be accepted more than one week after the original deadline, resulting in a mark of 0.",
    expected_output="If an assignment is submitted 3 days late, your mark will be reduced by 10%."
)

test2 = LLMTestCase(
    input="What happens if I submit my assignment 3 minutes late?",
    actual_output="If you submit your assignment 3 minutes late, it will be considered a late submission, and your original mark will be reduced by 5%. For example, if your original mark was 60%, it would be reduced to 57%. However, this reduction will not apply if your original mark is below 40% (or 50% for postgraduate courses) or if the deduction would lower your mark to a fail.",
    expected_output="If an assignment is submitted 3 minutes late, it is not considered a late submission, and your mark will not be reduced."
)

test3 = LLMTestCase(
    input="What is an EC claim?",
    actual_output="An EC claim, or extenuating circumstances claim, is a request made by students at Birmingham City University for consideration of circumstances that may have affected their academic performance. This can include reasons such as late submission of assessments, impaired performance during assessments, or absence from in-person assessments. The claim process allows students to seek support and potentially receive extensions or other accommodations based on their circumstances.",
    expected_output="An Extenuating Circumstances claim can be made by a student if there are circumstances that affect their ability to submit assessments on time, complete assessments to a good standard or attend in-person assessments."
)

test4 = LLMTestCase(
    input="What circumstances will be accepted as extenuating circumstances?",
    actual_output="Accepted extenuating circumstances include serious short-term illness or injury, worsening of an ongoing illness, death or significant illness of a close family member, unexpected caring responsibilities, and significant personal or family crises. Additionally, circumstances like witnessing a traumatic incident or an emergency that prevents you from attending an exam are also considered. However, it's important to provide appropriate evidence to support your claim. For more details, you can refer to the context provided.",
    expected_output="Serious short-term illness or injury, worsening of an ongoing illness or disability, symptoms of a harmful infectious disease, death or significant illness of a close family member or friend, unexpected caring responsibilities, significant personal or family crises, witnessing or experiencing a traumatic incident, a crime which has had a substantial impact on you, an accommodation crisis such as eviction."
)


test5 = LLMTestCase(
    input="When must I enrol?",
    actual_output="You must enrol at the start of your programme and for each level by the Friday of week four from the start date of your course.",
    expected_output="Students must enrol at the start of their programme and enrol for each level by the by the Friday of week four from the start date of their course unless a Break in Study has been approved."
)

test6 = LLMTestCase(
    input="What is the pass mark for a module?",
    actual_output="The pass mark for an undergraduate module is typically 40%, while for a postgraduate module, it is 50%. If you achieve a mark of 35-39% (UG) or 45-49% (PG) and meet certain criteria, you may be eligible for compensation for marginal failure.",
    expected_output="For an undergraduate course, the pass mark is 40%. On a postgraduate course, it is instead 50%."
)

test7 = LLMTestCase(
    input="What happens if I fail a module?",
    actual_output="If you fail a module, you may only be re-assessed in the failed assessment elements, as stated in the context: 'where a student fails a module, they may only be re-assessed in the failed assessment elements.' There are no limits on the number of modules or assessment components you can be reassessed in. If you pass the module after re-assessment, you will be awarded the module pass mark, but the overall mark will be capped at that pass mark. Additionally, failed modules do not contribute to the calculation of your overall mark required for progression or award classification.",
    expected_output="The first time you fail a module, you can be re-assessed for failed assessments, which is known as a resit. Your grade will be capped at the pass mark. You cannot be reassessed for assessments that you passed."
)

test8 = LLMTestCase(
    input="What are the degree classifications?",
    actual_output="I don't know. The retrieved context does not provide information about award classifications at Birmingham City University.",
    expected_output="Achieving an average of 70% or above grants you first-class honours. 60-69% is an upper second (2:1), 50-59% is a lower second (2:2), 40-49% is third-class honours, and anything below 40% is a fail."
)

test9 = LLMTestCase(
    input="How many BCU students are there?",
    actual_output="As of 2022-23, Birmingham City University (BCU) has a total of 31,300 students.",
    expected_output="BCU has 31,300 students as of 2022-23."
)

test10 = LLMTestCase(
    input="What is BCUSU?",
    actual_output="BCUSU, or Birmingham City University Student Union, is an independent charity that aims to improve students' university experiences. It represents student issues and concerns, working with the university to make positive changes. BCUSU is democratically run by students, with annual elections for SU positions. They offer advice on various topics, representation for student voices, and opportunities to join societies to connect with like-minded individuals.",
    expected_output="The Birmingham City University Student Union represent you as a student, and work together with the university to make change. They can help with various academic topics and offer societies to help people make friends."
)


# Create an evaluation dataset composed of the test cases.
dataset = EvaluationDataset(test_cases=[test1, test2, test3, test4, test5, test6, test7, test8, test9, test10])



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
evaluation_results = evaluate(dataset, metrics=[correctness_metric])

