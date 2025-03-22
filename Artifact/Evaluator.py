from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval import evaluate

# Define test cases with input, actual output, and expected output.
test_case1 = LLMTestCase(
    input="What happens if I submit my assignment 3 days late?",
    actual_output="If you submit your assignment 3 days (between 24 hours and one week) late, your original mark will be reduced by 10%. For example, if your original mark was 60%, it would be reduced to 54%. However, this reduction will not apply if your original mark is below 40% (or 50% for postgraduate courses) or if the deduction would lower your mark from a pass to a fail. Remember, work will not be accepted more than one week after the original deadline, resulting in a mark of 0.",
    expected_output="If an assignment is submitted 3 days late, your mark will be reduced by 10%."
)

test_case2 = LLMTestCase(
    input="What happens if I submit my assignment 3 minutes late?",
    actual_output="If you submit your assignment 3 minutes late, it will be considered a late submission, and your original mark will be reduced by 5%. For example, if your original mark was 60%, it would be reduced to 57%. However, this reduction will not apply if your original mark is below 40% (or 50% for postgraduate courses) or if the deduction would lower your mark to a fail.",
    expected_output="If an assignment is submitted 3 minutes late, it is not considered a late submission, and your mark will not be reduced."
)

# Create an evaluation dataset composed of the test cases.
dataset = EvaluationDataset(test_cases=[test_case1, test_case2])

# Initialize evaluation metrics.
answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)

correctness_metric = GEval(
    name="Correctness",
    criteria = "Determine whether the actual output is factually correct based on the expected output. Any additional scenarios or details are OK.",
    model="gpt-4o",  # Define the evaluation model
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ]
)

# Evaluate the dataset using the defined metrics.
evaluation_results = evaluate(dataset, metrics=[correctness_metric])