import json
import os

# تحديد مسارات الملفات
qrels_file_path = r"D:/iR/antique/test/qrels"
queries_file_path = r"D:/iR/antique/train/queries.txt"
output_file_path = 'D:/iR/antique/output.jsonl'

# قراءة محتوى ملف queries.txt
queries = {}
with open(queries_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        qid = parts[0]
        query = parts[1]
        queries[qid] = query

print("Loaded queries:", queries)

# قراءة محتوى ملف qrels وإنشاء البيانات المدمجة
output_data = []
with open(qrels_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        qid = parts[0]
        pid = parts[2].split('_')[0]
        if qid in queries:
            # البحث عن عنصر موجود بالفعل بنفس qid
            existing_entry = next((item for item in output_data if item["qid"] == int(qid)), None)
            if existing_entry:
                existing_entry["answer_pids"].append(int(pid))
            else:
                output_data.append({
                    "qid": int(qid),
                    "query": queries[qid],
                    "url": "",  # تحتاج إلى تعيين URL مناسب هنا
                    "answer_pids": [int(pid)]
                })

print("Output data before writing to file:", output_data)

# التحقق من الأذونات والكتابة إلى الملف إذا كان قابلاً للكتابة
if os.access(os.path.dirname(output_file_path), os.W_OK):
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')
    print(f"Data successfully written to {output_file_path}")
else:
    print(f"Permission denied: cannot write to {output_file_path}")
