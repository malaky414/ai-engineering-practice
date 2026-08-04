# 1. Dataset containing 10 test cases with Ground Truth expected values
TEST_CASES = [
    {
        "id": 1, 
        "text": "the program crashes when I open the report page, I need a fix immediately!",
        "expected": {"category": "Tech Support", "urgency": "High"}
    },
    {
        "id": 2, 
        "text": "what are your subscription prices for the new year?",
        "expected": {"category": "Sales", "urgency": "Low"}
    },
    {
        "id": 3, 
        "text": "the app is excellent and runs smoothly, thank you!",
        "expected": {"category": "Feedback", "urgency": "Low"}
    },
    {
        "id": 4, 
        "text": "the invoice has a discount that we didn't agree upon with sales, and I need it adjusted before I pay.",
        "expected": {"category": "Billing", "urgency": "High"}
    },
    {
        "id": 5, 
        "text": "I don't know how to change my password, and I've been trying since morning with no email received.",
        "expected": {"category": "Tech Support", "urgency": "Medium"}
    },
    {
        "id": 6, 
        "text": "I would like to export the files as a PDF.",
        "expected": {"category": "Feedback", "urgency": "Low"}
    },
    {
        "id": 7, 
        "text": "The amount was withdrawn from the bank account twice for the same service!",
        "expected": {"category": "Billing", "urgency": "High"}
    },
    {
        "id": 8, 
        "text": "There is a slight slowdown when I upload large files, but overall it works fine.",
        "expected": {"category": "Tech Support", "urgency": "Low"}
    },
    {
        "id": 9, 
        "text": "I need to cancel my account and get a refund if the technical issue isn't resolved.",
        "expected": {"category": "Billing", "urgency": "High"}
    },
    {
        "id": 10, 
        "text": "Can someone from the corporate department contact me for a new project?",
        "expected": {"category": "Sales", "urgency": "Medium"}
    }
]

# 2. Prompting Techniques

ZERO_SHOT_PROMPT = """
Classify the following customer query into:
- category: strictly one of [Tech Support, Billing, Sales, Feedback]
- urgency: strictly one of [Low, Medium, High]

You MUST return ONLY a valid JSON object with keys "category" and "urgency". No explanation or extra text.

Query: {text}
"""

FEW_SHOT_PROMPT = """
Classify the customer query into Category (Tech Support, Billing, Sales, Feedback) and Urgency (Low, Medium, High).
Return ONLY a valid JSON object: {{"category": "...", "urgency": "..."}}

Example 1:
Input: "I can't log into the system"
Output: {{"category": "Tech Support", "urgency": "High"}}

Example 2:
Input: "I need information about available pricing plans"
Output: {{"category": "Sales", "urgency": "Medium"}}

Example 3:
Input: "Thank you for the excellent service"
Output: {{"category": "Feedback", "urgency": "Low"}}

Input: {text}
Output:
Respond ONLY with the JSON object for the last input. Do not include any intro or markdown text.
"""

CHAIN_OF_THOUGHT_PROMPT = """
Classify the customer query into Category (Tech Support, Billing, Sales, Feedback) and Urgency (Low, Medium, High).

Analyze the input step-by-step first:
1. Identify the main intention/issue.
2. Determine how critical/time-sensitive it is.
3. Map to the appropriate category and urgency level.

Return ONLY a valid JSON object with keys: "reasoning", "category", "urgency".

Query: {text}
"""
