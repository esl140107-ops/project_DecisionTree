# Настройки компилятора
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O3 -Iinclude
LDFLAGS = -lm

# Структура проекта
SRC = src/main.c src/dataset.c src/decision_tree.c src/visualization.c
OBJ = $(SRC:.c=.o)
TARGET = cart_app.exe

# Основные правила сборки 

all: $(TARGET)

$(TARGET): $(OBJ)
	@echo [Build] Linking executable: $(TARGET)...
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	@echo [Compile] Compiling source file: $<...
	$(CC) $(CFLAGS) -c $< -o $@

# --- ГЛОБАЛЬНЫЙ ТЕСТ (ЗАПУСК ВСЕГО ПАКЕТА) ---

test_all: iris diabetes house
	@echo.
	@echo =========================================================
	@echo   ALL TESTS COMPLETED AND VISUALIZED!
	@echo =========================================================
	@echo.

#Индивидуальные тесты

iris: $(TARGET)
	@echo.
	@echo === [1/3] Running: Iris Classification (Entropy) ===
	./$(TARGET) iris.csv 5 c e
	-dot -Tpng tree.dot -o tree_iris.png
	-code tree_iris.png

diabetes: $(TARGET)
	@echo.
	@echo === [2/3] Running: Diabetes Risk Prediction (Gini) ===
	./$(TARGET) diabetes.csv 5 c g
	-dot -Tpng tree.dot -o tree_diabetes.png
	-code tree_diabetes.png

house: $(TARGET)
	@echo.
	@echo === [3/3] Running: House Price Analysis (MSE) ===
	./$(TARGET) house_prices.csv 10 r m
	-dot -Tpng tree.dot -o tree_house.png
	-code tree_house.png

#Очистка

clean:
	@echo [Clean] Removing temporary files and artifacts...
	-if exist src\*.o del /q src\*.o
	-if exist $(TARGET) del /q $(TARGET)
	-if exist tree.dot del /q tree.dot
	-if exist tree_*.png del /q tree_*.png

.PHONY: all clean iris diabetes house test_all
