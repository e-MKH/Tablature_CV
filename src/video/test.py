from ultralytics import YOLO

# 모델 로드
model = YOLO('weights.pt')

# confidence threshold를 낮춰서 테스트
results = model.predict('image.png', conf=0.1, verbose=False)

# 결과 처리
result = results[0]
class_names = result.names

fret_count = 0
string_count = 0

# 한 번의 loop에서 카운트와 confidence 출력 동시에
for box in result.boxes:
    cls_name = class_names[int(box.cls[0])]
    confidence = float(box.conf[0])
    print(f"{cls_name}: {confidence:.3f}")

    if cls_name == 'fret':
        fret_count += 1
    elif cls_name == 'string':
        string_count += 1

# 최종 카운트 출력
print(f"\n프렛(fret): {fret_count}개")
print(f"현(string): {string_count}개")