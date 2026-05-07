# from rag_engine import NileTelRAG

# rag = NileTelRAG(data_dir="data", cache_dir="cache")

# # Test cases covering all routing scenarios
# tests = [
#     "ازيك",                                         # pure greeting
#     "مرحبا، ايه باقات الإنترنت؟",                  # greeting + chat
#     "ابعتلي مهندس",                                # ticket
#     "مرحبا، انقطع النت عندي",                     # greeting + ticket
#     "ايه الـ SLA وكمان ايه باقة 100 ميجا؟",       # multi-question chat
#     "ايه أحسن مطعم في القاهرة؟",                  # out of scope
# ]

# for q in tests:
#     print("\n" + "="*60)
#     print(f"Q: {q}")
#     result = rag.query(q)
#     print(f"Route : {result['route']}")
#     print(f"Action: {result['needs_action']}")
#     print(f"Answer: {result['answer']}")

from rag_engine import NileTelRAG

rag = NileTelRAG(data_dir="data", cache_dir="cache")

tests = [
    "ازيك",                                        # pure greeting
    "مرحبا، ايه باقات الإنترنت؟",                 # greeting + chat
    "ابعتلي مهندس",                               # ticket
    "مرحبا، انقطع النت عندي",                     # greeting + ticket
    "ايه الـ SLA وكمان ايه باقة 100 ميجا؟",      # multi-question chat
    "ايه أحسن مطعم في القاهرة؟",                 # out of scope
]

output_lines = []

for q in tests:
    print(f"Running: {q}")
    result = rag.query(q)

    block = (
        "\n" + "="*60 + "\n"
        f"Q      : {q}\n"
        f"Route  : {result['route']}\n"
        f"Action : {result['needs_action']}\n"
        f"Sources: {', '.join(result['sources']) if result['sources'] else 'none'}\n"
        f"Answer :\n{result['answer']}\n"
    )
    output_lines.append(block)
    print(block)

with open("results.txt", "w", encoding="utf-8") as f:
    f.writelines(output_lines)

print("\nResults saved to results.txt")