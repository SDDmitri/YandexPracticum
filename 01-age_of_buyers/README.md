# Определение возраста покупателей по фотографии

## Описание проекта
Требуется построить модель, которая по фотографии определит приблизительный возраст клиента. Эта модель позволит:
- анализировать покупки и предлагать товары, которые могут заинтересовать покупателей этой возрастной группы
- контролировать добросовестность кассиров при продаже алкоголя
Для обучения модели доступны фотографии людей с указанием возраста.

## Навыки и инструменты
- Python
- sys
- Pandas
- Matplotlib
- TensorFlow
- Keras

## Выводы
1. в качестве модели использовали предобученную `ResNet50` на датасете `ImageNet`, к которой добавили три слоя: GlobalAveragePooling2D и два полносвязанных слоя на 50 и 1 нейрон с функцией активации ReLu
1. над обучающей выборкой провели аугментацию: отражение по горизонтали, поворот в пределах 45 градусов, сдвиг по ширине и высоте на 20%, нормализация яркости, масштабирование
1. все фотографии для обучения привели к размеру 224x224 