# Vulkan C++ application

<img src="./screenshots/screenshot1.PNG" width="1000px"> <img src="./screenshots/screenshot2.PNG" width="1000px"> <img src="./screenshots/screenshot3.PNG" width="1000px">

## Что это

Данное приложение осуществляет рендеринг сцены с помощью Vulkan API. Основные файлы с кодом лежат в папке core, сторонние библиотеки лежат в папке libs, дополнительные ресурсы в виде моделей и текстур находятся в папках model и texture соответственно. Проект выполнен в Qt Creator.

## Что реализовано

* Базовая отрисовка 3D геометрии. Текстурирование.
* Освещение прожекторными и точечными источниками.
* Мягкие тени.
* Отрисовка в различные цветовые прикрепления.
* Многопроходный рендеринг. В частности осуществляется 2 прохода - в первом осуществляется отрисовка моделей, во втором - работа над изображением, например, размытие по Гауссу.
* Рендер glTF моделей на основе [загрузчика glTF](https://github.com/SaschaWillems/Vulkan-glTF-PBR), переработанного под данную реализацию приложения Vulkan.
* Анимация. Линейная интерполяция между анимациями (На картинке передняя пчела плавно меняет свою анимацию на анимацию полёта, как у задней пчелы, и обратно)
<img src="./screenshots/Vulkan.gif" width="1000px">

* Группировка объектов (в том числе источников света и камеры) и управление группами.
* Поддержка нескольких буферов кадра
* Мультисемплинг.
* Создание MIP-карт.
* Скайбокс

* Отложенный рендеринг (за счёт подпроходов рендера)

<img src="./screenshots/screenshot5.PNG" width="400px"> <img src="./screenshots/screenshot6.PNG" width="400px">
<img src="./screenshots/screenshot7.PNG" width="400px"> <img src="./screenshots/screenshot8.PNG" width="400px">

* Объёмный свет
<img src="./screenshots/screenshot4.PNG" width="1000px">

* Screen Space Local Reflections
<img src="./screenshots/screenshot9.PNG" width="1000px">

* Использование трафаретного буфера для выделения объектов (возможность включать и выключать выделение) 
<img src="./screenshots/stencil.gif" width="1000px">

* Прототип обьёмных источников света (сферические источники, плоскости)
<img src="./screenshots/areaLight.gif" width="1000px">

* Буфер, который можно считывать со стороны CPU, благодаря чему можно распознавать объекты под курсором
* Простая физика объектов
<img src="./screenshots/collisions.gif" width="1000px">
