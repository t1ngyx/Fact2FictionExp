from infact.eval.evaluate import evaluate
from multiprocessing import set_start_method
import argparse

if __name__ == '__main__':  # evaluation uses multiprocessing
    set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--procedure_variant", type=str, default="no_qa")
    parser.add_argument("--llm", type=str, default="llama33_70b_free")
    parser.add_argument("--variant", type=str, default="dev")
    args = parser.parse_args()

    evaluate(
        llm=args.llm,
        tools_config=dict(searcher=dict(
            search_engine_config=dict(
                averitec_kb=dict(variant=args.variant),
            ),
            limit_per_search=5
        )),
        fact_checker_kwargs=dict(
            procedure_variant=args.procedure_variant,
            max_iterations=3,
            max_result_len=64_000,  # characters
        ),
        llm_kwargs=dict(temperature=0.01),
        benchmark_name="averitec",
        benchmark_kwargs=dict(variant=args.variant),
        random_sampling=False,
        print_log_level="info",
        n_workers=4, 
    )


# from infact.eval.evaluate import evaluate
# from multiprocessing import set_start_method

# variant = "dev"

# if __name__ == '__main__':  # evaluation uses multiprocessing
#     set_start_method("spawn")
#     evaluate(
#         llm="gpt_4o_mini",
#         tools_config=dict(searcher=dict(
#             search_engine_config=dict(
#                 averitec_kb=dict(variant=variant),
#             ),
#             limit_per_search=5
#         )),
#         fact_checker_kwargs=dict(
#             procedure_variant="no_qa",
#             max_iterations=3,
#             max_result_len=64_000,  # characters
#         ),
#         llm_kwargs=dict(temperature=0.01),
#         benchmark_name="averitec",
#         benchmark_kwargs=dict(variant=variant),
#         random_sampling=False,
#         print_log_level="info",
#         n_workers=4,
#     )
