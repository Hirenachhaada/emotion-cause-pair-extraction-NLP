import csv

def parse_data_with_causing_statements(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            scene_info = lines[i].strip().split()
            scene_id, num_conversations = int(scene_info[0]), int(scene_info[1])
            i += 1
            pairs = lines[i].strip()
            causing_map = {}
            if pairs != '':
                for pair in pairs.strip('()').split('),('):
                    effect, cause = map(int, pair.split(','))
                    if effect not in causing_map:
                        causing_map[effect] = []
                    causing_map[effect].append(cause)
            i += 1
            for _ in range(num_conversations):
                parts = lines[i].strip().split('|')
                conv_id = int(parts[0].strip())
                data.append({
                    'scene_id': scene_id,
                    'conversation_id': conv_id,
                    'causing_pairs': causing_map.get(conv_id, []),
                    'speaker': parts[1].strip(),
                    'emotion': parts[2].strip(),
                    'statement': parts[3].strip(),
                    'episode_info': parts[4].strip(),
                })
                i += 1
    return data

def write_to_csv(data, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['scene_id', 'conversation_id', 'causing_pairs', 'speaker', 'emotion', 'statement', 'episode_info']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

# Usage
input_file = 'test.txt'  # Replace with your input file path
output_file = 'test.csv'  # Replace with your desired output file path

data = parse_data_with_causing_statements(input_file)
write_to_csv(data, output_file)

print(f"CSV file has been created: {output_file}")
