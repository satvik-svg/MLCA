from __future__ import annotations

from typing import Dict, List

RESOURCE_MAP: Dict[str, Dict[str, List[str]]] = {
    "Mathematics": {
        "core": [
            "Algebra Foundations",
            "Statistics Essentials",
            "Problem Solving with Practice Sets",
            "Quantitative Reasoning Basics",
        ],
        "STEM": [
            "Applied Calculus for Beginners",
            "Data Interpretation for Science Projects",
        ],
        "Commerce": [
            "Business Math and Financial Numeracy",
            "Excel for Quantitative Analysis",
        ],
    },
    "Science": {
        "core": [
            "Conceptual Physics and Everyday Science",
            "Core Biology Revision Program",
            "Chemistry Through Visual Experiments",
            "Scientific Thinking and Lab Skills",
        ],
        "STEM": [
            "Intro to Scientific Computing",
            "Research Methods for School Students",
        ],
        "Language": [
            "Science Communication Skills",
            "Read and Explain Scientific Articles",
        ],
    },
    "English": {
        "core": [
            "Grammar Mastery Bootcamp",
            "Academic Writing and Structure",
            "Reading Comprehension Accelerator",
            "Vocabulary and Expression Builder",
        ],
        "Arts": [
            "Creative Writing Workshop",
            "Storytelling and Literary Analysis",
        ],
        "Language": [
            "Advanced Communication Skills",
            "Public Speaking for Students",
        ],
    },
}


def _deduplicate(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def get_recommendations(weak_subject: str, interest: str) -> List[str]:
    subject_payload = RESOURCE_MAP.get(weak_subject)
    if not subject_payload:
        return ["General Study Skills Course", "Weekly Revision Planning Guide"]

    recommendations = list(subject_payload.get("core", []))
    recommendations.extend(subject_payload.get(interest, []))
    return _deduplicate(recommendations)


def get_study_tips(weak_subject: str, study_time_hours: float, marks: dict) -> List[str]:
    tips = [
        "Use 45-minute focused sessions followed by 10-minute breaks.",
        "Review mistakes weekly and maintain an error log.",
    ]

    if study_time_hours < 2.0:
        tips.append("Increase study time gradually to at least 2.5 hours/day.")
    elif study_time_hours < 4.0:
        tips.append("Try one additional 30-minute revision block each day.")

    weak_mark = marks.get(weak_subject, 0)
    if weak_mark < 50:
        tips.append("Start with basics and solve at least 10 foundational questions daily.")
    elif weak_mark < 65:
        tips.append("Focus on medium-difficulty practice and timed quizzes.")
    else:
        tips.append("Maintain consistency with mixed-level practice and weekly mock tests.")

    if weak_subject == "Mathematics":
        tips.append("Practice step-by-step solutions and check each calculation carefully.")
    elif weak_subject == "Science":
        tips.append("Use diagrams and concept maps to connect formulas and processes.")
    elif weak_subject == "English":
        tips.append("Read one article daily and summarize it in your own words.")

    return tips
