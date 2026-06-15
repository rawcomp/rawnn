CC      := gcc
CFLAGS  := -Wall -Wextra -Werror -pedantic -std=c99 -I./include \
           -Wshadow -Wformat=2  -Wstrict-prototypes  \
           -Wmissing-prototypes -Wnull-dereference                \
            -Wundef -Wwrite-strings             \
           -Wfloat-equal -Wcast-qual -Wpointer-arith              \
           -fstack-protector-strong -fno-common
LDFLAGS := -lm
VALGRIND := valgrind --leak-check=full --show-leak-kinds=all --errors-for-leak-kinds=all --error-exitcode=1

PREFIX ?= /usr/local
BUILD_DIR := build

LIB_NAME  := $(BUILD_DIR)/rawcompute.a

SRC_DIR   := src
TEST_DIR  := tests
INC_DIR   := include

SRC_FILES := $(wildcard $(SRC_DIR)/r_*.c)
SRC_OBJS  := $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/$(SRC_DIR)/%.o, $(SRC_FILES))

TEST_SRCS := $(wildcard $(TEST_DIR)/test_*.c)
TEST_OBJS := $(patsubst $(TEST_DIR)/%.c, $(BUILD_DIR)/$(TEST_DIR)/%.o, $(TEST_SRCS))
TESTS     := $(patsubst $(TEST_DIR)/%.c, $(BUILD_DIR)/$(TEST_DIR)/%, $(TEST_SRCS))

HEADERS   := $(wildcard $(INC_DIR)/*.h) $(wildcard $(INC_DIR)/rc/*.h)

.PHONY: all test valgrind clean format install

all: $(LIB_NAME)

$(LIB_NAME): $(SRC_OBJS)
	@mkdir -p $(dir $@)
	@echo "---- Building library $(notdir $@) ----"
	ar rcs $@ $^

$(BUILD_DIR)/$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

test: $(LIB_NAME) $(TESTS)
	@echo "---- Running all tests ----"
	@for exe in $(TESTS); do \
		log_file="$$exe.log"; \
		"$$exe" > "$$log_file" 2>&1; \
		status=$$?; \
		if [ $$status -eq 0 ]; then \
			echo "PASS: $$(basename $$exe)"; \
		else \
			echo "FAIL: $$(basename $$exe)"; \
			cat "$$log_file"; \
			exit $$status; \
		fi \
	done
	@echo "---- All tests completed ----"

$(BUILD_DIR)/$(TEST_DIR)/%: $(BUILD_DIR)/$(TEST_DIR)/%.o $(LIB_NAME)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $< $(LIB_NAME) -o $@ $(LDFLAGS)

$(BUILD_DIR)/$(TEST_DIR)/%.o: $(TEST_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

valgrind: test
	@echo "---- Starting Valgrind checks ----"
	@for exe in $(TESTS); do \
		log_file="$$exe.valgrind.log"; \
		$(VALGRIND) "$$exe" > "$$log_file" 2>&1; \
		status=$$?; \
		if [ $$status -eq 0 ]; then \
			echo "VALGRIND PASS: $$(basename $$exe)"; \
		else \
			echo "VALGRIND FAIL: $$(basename $$exe)"; \
			cat "$$log_file"; \
			exit $$status; \
		fi \
	done

format:
	@echo "---- Formatting files with clang-format (Microsoft style) ----"
	clang-format -i --style=Microsoft $(SRC_FILES) $(TEST_SRCS) $(HEADERS)

install: $(LIB_NAME)
	@echo "---- Installing library and headers ----"
	@mkdir -p $(DESTDIR)$(PREFIX)/lib
	@mkdir -p $(DESTDIR)$(PREFIX)/include/rc
	cp $(LIB_NAME) $(DESTDIR)$(PREFIX)/lib/
	cp $(wildcard $(INC_DIR)/rc/*.h) $(DESTDIR)$(PREFIX)/include/rc/
	@if [ -n "$$(ls -A $(INC_DIR)/*.h 2>/dev/null)" ]; then \
		cp $(INC_DIR)/*.h $(DESTDIR)$(PREFIX)/include/; \
	fi
	@echo "---- Install completed ----"

clean:
	rm -rf $(BUILD_DIR)
	@echo "---- Clean completed ----"
