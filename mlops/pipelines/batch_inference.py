from rag import generate_rag_response

def run_batch(queries):
    results = []

    for q in queries:
        res = generate_rag_response(q)
        results.append(res)

    return results


if __name__ == "__main__":
    sample = [
        "water leakage in street",
        "road pothole issue",
        "electricity cut in area"
    ]

    print(run_batch(sample))