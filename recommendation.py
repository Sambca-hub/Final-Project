"""Rule-based learning recommendations from marks, attendance, and subject."""


def recommend_resources(subject, marks, attendance):
    """Return practical recommendations; this is a rules-based system, not a trained AI model."""
    try:
        marks = float(marks)
        attendance = float(attendance)
    except (TypeError, ValueError):
        raise ValueError("Marks and attendance must be numeric.") from None

    if not 0 <= marks <= 100:
        raise ValueError("Marks must be between 0 and 100.")
    if not 0 <= attendance <= 100:
        raise ValueError("Attendance must be between 0 and 100.")

    normalized_subject = " ".join(str(subject or "").strip().split()).casefold()
    subject_resources = {
        "database": ["Review SQL fundamentals and relational database concepts.", "Practice SELECT, JOIN, GROUP BY, and query-writing exercises."],
        "python": ["Review Python fundamentals, functions, and data structures.", "Build a small Python project and practice debugging."],
        "mathematics": ["Review the formulas and concepts used in recent lessons.", "Work through practice problems, checking each solution step."],
        "computer networks": ["Review networking fundamentals, protocols, and OSI/TCP-IP layers.", "Practice subnetting and basic network configuration exercises."],
    }

    recommendations = []
    if marks < 50:
        recommendations.extend([
            "Start with the foundational concepts for this subject.",
            "Practice short, guided exercises and review mistakes with your instructor.",
        ])
    elif marks < 70:
        recommendations.extend([
            "Revise topics where you lost marks and practice mixed exercises.",
            "Use worked examples, then solve similar questions independently.",
        ])
    else:
        recommendations.extend([
            "Explore advanced topics and apply concepts in a small project.",
            "Try challenging questions and explain your solution approach.",
        ])

    if attendance < 75:
        recommendations.append(
            "Attendance is below 75%; plan to attend upcoming classes and review missed lessons."
        )

    recommendations.extend(subject_resources.get(normalized_subject, [
        f"Review the core concepts and terminology in {str(subject).strip() or 'this subject'}.",
        "Practice additional exercises and use your course notes to check your answers.",
    ]))
    return recommendations
