import json
import argparse
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset, Dataset

def get_dataset():
    ds = load_dataset("AlekseyKorshuk/quora-question-pairs")
    ds_data = ds['train']
    return ds_data

def get_json(run_id, path: str):
    file_path = f"{path}/cache_run_{run_id}.json"
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_to_json(data, file_name):
    dict_data = {}
    for i, entry in enumerate(data):
        dict_data[i] = {
            "id":               entry[0],
            "question_1":       entry[1][0],
            "event_type_1":     entry[1][1],
            "question_2":       entry[2][0],
            "event_type_2":     entry[2][1],
            "is_duplicate":     entry[3]
        }
    
    with open(f"quora_qp_results/{file_name}.json", 'w') as f:
        json.dump(dict_data, f, indent=4)

def calculate_percentage(check, part):
    return (part/check)*100

def print_percentages(check, 
                          all_duplicates,
                          false_negatives,
                          positive_hits,
                          positive_miss,
                          false_positive,
                          ambiguous,
                          negative_routing,
                          positive_routing):
    print(f"all_duplicates: {calculate_percentage(check, all_duplicates):.2f}%")
    print(f"\tfalse_negatives:  {calculate_percentage(check, false_negatives):.2f}%")
    print(f"\tpositive_hits:    {calculate_percentage(check, positive_hits):.2f}%")
    print(f"positive_miss:  {calculate_percentage(check, positive_miss):.2f}%")
    print(f"false_positive: {calculate_percentage(check, false_positive):.2f}%")
    print(f"ambiguous:      {calculate_percentage(check, ambiguous):.2f}%")
    print(f"negative_routing:      {calculate_percentage(check, negative_routing):.2f}%")
    print(f"positive_routing:      {calculate_percentage(check, positive_routing):.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Parse arguments for quore qp dataset evaluation")
    parser.add_argument('--path', type=str, required=True, help='Path to run data json')
    parser.add_argument('--run_id', type=str, required=True, help='ID for run data')
    args = parser.parse_args()

    ds = get_dataset()

    print(args.run_id)
    print(args.path)

    run_data = get_json(args.run_id, args.path)
    
    requests = run_data['requests']
    req_num = len(requests)
    print(req_num)

#    for i in range(0, 10, 1):
#        print(ds[i])
#    for i in range(990, 1000, 1):
#        print(ds[i])

    temp_results = []
    # dict that keeps track of request in which questions were asked.
    # Organization:
    #    qid: req_id
    question_use = {}

    for i in range(0, req_num, 2):
        q1 = [requests[i]['query'],requests[i]['event_type'], requests[i]['response']]
        q2 = [requests[i+1]['query'],requests[i+1]['event_type'], requests[i]['response']]
        ds_i = i//2
        qid1 = ds[ds_i]['qid1']
        qid2 = ds[ds_i]['qid2']
        temp = [ds_i, q1, q2, ds[ds_i]['is_duplicate'], qid1, qid2]
        question_use.setdefault(qid1, []).append(i)
        question_use.setdefault(qid2, []).append(i+1)
        temp_results.append(temp)

    all_duplicates      = []
    false_negatives     = []
    positive_hits       = []
    positive_miss       = []
    ambiguous           = []
    false_positive      = []
    negative_routing    = []
    positive_routing    = []
    for r in temp_results:
        q1 = r[1]
        q2 = r[2]
        control = r[3]

        if control == 1:
            all_duplicates.append(r)

        if q2[1] == "CACHE_MISS" and control == 1:                             # a
            false_negatives.append(r)
        if q2[1] == "CACHE_HIT" and control == 1:                              # b
            positive_hits.append(r)
        if q2[1] == "CACHE_MISS" and control == 0:                             # c
            positive_miss.append(r)
        if q2[1] == "CACHE_HIT" and control == 0:                              # d
            if q1[2] == q2[2]:                        # This is definitly a false positive
                false_positive.append(r)
            else:
                ambiguous.append(r)
        if q2[1] == "LLM_DIRECT_CALL" and control == 1:                        # e
            negative_routing.append(r)
        if q2[1] == "LLM_DIRECT_CALL" and control == 0:                        # f
            positive_routing.append(r)

    len_all_duplicates      = len(all_duplicates)
    len_false_negatives     = len(false_negatives)
    len_positive_hits       = len(positive_hits)
    len_positive_miss       = len(positive_miss)
    len_ambiguous           = len(ambiguous)
    len_false_positive      = len(false_positive)
    len_negative_routing    = len(negative_routing)
    len_positive_routing    = len(positive_routing)

    check = len_false_negatives + len_positive_hits + len_positive_miss + len_false_positive + len_ambiguous + len_negative_routing + len_positive_routing

    print('=' * 60)
    print(f"CHECK: {check}")
    print(f'temp_results: {len(temp_results)}')
    print(f'all_duplicates: {len_all_duplicates}')
    print(f'false_negatives: {len_false_negatives}')
    print(f'positive_hits: {len_positive_hits}')
    print(f'positive_miss: {len_positive_miss}')
    print(f'false_positive: {len_false_positive}')
    print(f'ambiguous: {len_ambiguous}')
    print(f'negative_routing: {len_negative_routing}')
    print(f'positive_routing: {len_negative_routing}')
    print('=' * 60)

    print_percentages(check, 
                          len_all_duplicates,
                          len_false_negatives,
                          len_positive_hits,
                          len_positive_miss,
                          len_false_positive,
                          len_ambiguous,
                          len_negative_routing,
                          len_positive_routing)
    
    print('=' * 60)

    write_to_json(all_duplicates, "all_duplicates")
    write_to_json(false_negatives, "false_negatives")
    write_to_json(positive_hits, "positive_hits")
    write_to_json(positive_miss, "positive_miss")
    write_to_json(false_positive, "false_positive")
    write_to_json(ambiguous, "ambiguous")

if __name__=="__main__":
    main()