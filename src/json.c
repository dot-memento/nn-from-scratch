/**
 * @file json.c
 * @brief Implementation of a lightweight JSON parser and manipulation library.
 *
 * @author Michael Teixeira
 * @copyright MIT License
 * @see https://github.com/dot-memento/json-lib
 */

#include "json.h"

#include <stdint.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdarg.h>

#define CHECK_TYPE(entry, expected_type) if ((entry)->type != (expected_type)) return JSON_ERROR_WRONG_TYPE

static const size_t DEFAULT_MAX_DEPTH = 1000;
static const size_t INITIAL_STRING_BUFFER_SIZE = 16;

#if !defined(_POSIX_C_SOURCE) && !defined(_DEFAULT_SOURCE) && \
    !defined(_BSD_SOURCE) && !defined(_SVID_SOURCE) && \
    (__STDC_VERSION__ < 202311L)

// Duplicates a string using dynamic allocation.
// Only define strdup if not available in the standard library.
static char *strdup(const char *string)
{
    size_t size = strlen(string);
    char *string_copy = malloc(size + 1);
    return string_copy ? strcpy(string_copy, string) : NULL;
}

#endif

// --------------------
// Dynamic String Builder
// --------------------

// Structure to build strings dynamically.
typedef struct string_builder {
    size_t allocated_size;
    size_t size;
    char *data;
} string_builder;

// Frees resources allocated by the string builder.
static void string_builder_free(string_builder *builder)
{
    free(builder->data);
    builder->allocated_size = 0;
    builder->size = 0;
    builder->data = NULL;
}

// Ensures the builder has enough capacity; double size when needed.
static bool string_builder_ensure_capacity(string_builder *builder, size_t min_capacity)
{
    if (min_capacity + 1 > builder->allocated_size)
    {
        char *new_data;
        if (builder->allocated_size == 0)
        {
            builder->allocated_size = INITIAL_STRING_BUFFER_SIZE;
            new_data = malloc(builder->allocated_size);
        }
        else
        {
            builder->allocated_size *= 2;
            new_data = realloc(builder->data, builder->allocated_size);
        }
        if (!new_data)
            return true;
        builder->data = new_data;
    }
    return false;
}

// Appends a single character to the string builder.
static bool string_builder_append(string_builder *builder, char c)
{
    if (string_builder_ensure_capacity(builder, builder->size + 1)) return true;

    builder->data[builder->size++] = c;
    builder->data[builder->size] = '\0';
    return false;
}

// Appends a single character to the string builder.
static bool string_builder_append_string(string_builder *builder, const char *str)
{
    size_t length = strlen(str);
    if (string_builder_ensure_capacity(builder, builder->size + length)) return true;

    memcpy(builder->data + builder->size, str, length);
    builder->size += length;
    builder->data[builder->size] = '\0';
    return false;
}

static bool string_builder_append_format(string_builder *builder, const char *format, va_list args)
{
    va_list args_copy;

    va_copy(args_copy, args);
    int length = vsnprintf(NULL, 0, format, args_copy);
    va_end(args_copy);

    if (length < 0 || string_builder_ensure_capacity(builder, builder->size + length))
    {
        va_end(args);
        return true;
    }

    vsprintf(builder->data + builder->size, format, args);
    builder->size += length;

    return false;
}

// Appends a UTF code point (encoded in UTF-8) to the builder.
static bool string_builder_append_utf_code_point(string_builder *builder, uint32_t code_point)
{
    if (string_builder_ensure_capacity(builder, builder->size + 4)) return true;
   
    if (code_point <= 0x7F)
        builder->data[builder->size++] = code_point & 0xFF;
    else if (code_point <= 0x7FF)
    {
        builder->data[builder->size++] = 0xC0 | ((code_point >> 6) & 0x1F);
        builder->data[builder->size++] = 0x80 | (code_point & 0x3F);
    }
    else if (code_point <= 0xFFFF)
    {
        builder->data[builder->size++] = 0xE0 | ((code_point >> 12) & 0x0F);
        builder->data[builder->size++] = 0x80 | ((code_point >> 6) & 0x3F);
        builder->data[builder->size++] = 0x80 | (code_point & 0x3F);
    }
    else if (code_point <= 0x10FFFF)
    {
        builder->data[builder->size++] = 0xF0 | ((code_point >> 18) & 0x07);
        builder->data[builder->size++] = 0x80 | ((code_point >> 12) & 0x3F);
        builder->data[builder->size++] = 0x80 | ((code_point >> 6) & 0x3F);
        builder->data[builder->size++] = 0x80 | (code_point & 0x3F);
    }
    else
    {
        // Invalid code point, replace with replacement character (U+FFFD).
        builder->data[builder->size++] = (char)0xEF;
        builder->data[builder->size++] = (char)0xBF;
        builder->data[builder->size++] = (char)0xBD;
    }
    builder->data[builder->size] = '\0';

    return false;
}

// Finalizes and builds the string from the builder.
static bool string_builder_build(string_builder *builder, char **out)
{
    char *string = realloc(builder->data, builder->size + 1);
    if (!string) return true;

    string[builder->size] = '\0';
    builder->allocated_size = 0;
    builder->size = 0;
    builder->data = NULL;
    *out = string;
    return false;
}

// --------------------
// JSON Error Handling
// --------------------

const char* json_error_to_string(json_error code)
{
    switch (code)
    {
    case JSON_SUCCESS: return "success";
    case JSON_ERROR_ALLOCATION: return "allocation error";
    case JSON_ERROR_NULL: return "null pointer argument";
    case JSON_ERROR_WRONG_TYPE: return "wrong type";
    case JSON_ERROR_INDEX_OUT_OF_BOUNDS: return "index out of bounds";
    case JSON_ERROR_KEY_NOT_FOUND: return "key not found";
    case JSON_ERROR_IO: return "I/O error";
    case JSON_ERROR_INVALID_OPTIONS: return "invalid options";
    case JSON_ERROR_MAX_DEPTH: return "maximum depth exceeded";
    case JSON_ERROR_NUMBER_FORMAT: return "invalid number format";
    case JSON_ERROR_ESCAPE_SEQUENCE: return "invalid string escape";
    case JSON_ERROR_UNICODE: return "invalid unicode sequence";
    case JSON_ERROR_BUFFER_TOO_SMALL: return "buffer too small";
    case JSON_ERROR_CIRCULAR_REFERENCE: return "circular reference";
    default: return "unknown error";
    }
}

// --------------------
// JSON Data Structures
// --------------------

typedef struct json_array {
    size_t length;
    json_value *entry[];
} json_array;

typedef struct json_object {
    size_t size;
    char **keys;
    json_value *entry[];
} json_object;

typedef struct json_value {
    json_type type;
    union {
        double number;
        char *string;
        bool boolean;
        json_array *array;
        json_object *object;
    };
} json_value;

// --------------------
// JSON Creation API
// --------------------

json_error json_null_create(json_value **out)
{
    if (!out) return JSON_ERROR_NULL;

    json_value *entry = malloc(sizeof(json_value));
    if (!entry) return JSON_ERROR_ALLOCATION;

    *entry = (json_value) {0};
    entry->type = JSON_NULL;
    *out = entry;
    return JSON_SUCCESS;
}

json_error json_bool_create(bool value, json_value **out)
{
    if (!out) return JSON_ERROR_NULL;

    json_value *entry = malloc(sizeof(json_value));
    if (!entry) return JSON_ERROR_ALLOCATION;

    *entry = (json_value) {0};
    entry->type = JSON_BOOL;
    entry->boolean = value;
    *out = entry;
    return JSON_SUCCESS;
}

json_error json_number_create(double value, json_value **out)
{
    if (!out) return JSON_ERROR_NULL;

    json_value *entry = malloc(sizeof(json_value));
    if (!entry) return JSON_ERROR_ALLOCATION;

    *entry = (json_value) {0};
    entry->type = JSON_NUMBER;
    entry->number = value;
    *out = entry;
    return JSON_SUCCESS;
}

json_error json_string_create(const char *value, json_value **out)
{
    if (!value || !out) return JSON_ERROR_NULL;

    char *string_copy = strdup(value);
    if (!string_copy) return JSON_ERROR_ALLOCATION;

    json_value *entry = malloc(sizeof(json_value));
    if (!entry)
    {
        free(string_copy);
        return JSON_ERROR_ALLOCATION;
    }
    *entry = (json_value) {0};
    entry->type = JSON_STRING;
    entry->string = string_copy;
    *out = entry;
    return JSON_SUCCESS;
}

json_error json_string_create_nocopy(char *value, json_value **out)
{
    if (!value || !out) return JSON_ERROR_NULL;

    json_value *entry = malloc(sizeof(json_value));
    if (!entry) return JSON_ERROR_ALLOCATION;

    *entry = (json_value) {0};
    entry->type = JSON_STRING;
    entry->string = value;
    *out = entry;
    return JSON_SUCCESS;
}

json_error json_array_create(json_value **out)
{
    if (!out) return JSON_ERROR_NULL;

    json_value *entry = malloc(sizeof(json_value));
    if (!entry) return JSON_ERROR_ALLOCATION;

    json_array *array = malloc(sizeof(json_array));
    if (!array)
    {
        free(entry);
        return JSON_ERROR_ALLOCATION;
    }

    *entry = (json_value) {0};
    entry->type = JSON_ARRAY;
    entry->array = array;
    array->length = 0;
    *out = entry;
    return JSON_SUCCESS;
}

json_error json_object_create(json_value **out)
{
    if (!out) return JSON_ERROR_NULL;

    json_value *entry = malloc(sizeof(json_value));
    if (!entry) return JSON_ERROR_ALLOCATION;

    json_object *object = malloc(sizeof(json_object));
    if (!object)
    {
        free(entry);
        return JSON_ERROR_ALLOCATION;
    }

    object->size = 0;
    object->keys = NULL;
    *entry = (json_value) {0};
    entry->type = JSON_OBJECT;
    entry->object = object;
    *out = entry;
    return JSON_SUCCESS;
}

json_error json_clone(const json_value *entry, json_value **out)
{
    switch (entry->type)
    {
    case JSON_NULL:   return json_null_create(out);
    case JSON_BOOL:   return json_bool_create(entry->boolean, out);
    case JSON_NUMBER: return json_number_create(entry->number, out);
    case JSON_STRING: return json_string_create(entry->string, out);

    case JSON_ARRAY:
    {
        json_value *new_entry = malloc(sizeof(json_value));
        if (!new_entry) return JSON_ERROR_ALLOCATION;

        json_array *new_array = malloc(sizeof(json_array) + entry->array->length * sizeof(json_value*));
        if (!new_array)
        {
            free(new_entry);
            return JSON_ERROR_ALLOCATION;
        }

        new_entry->type = JSON_ARRAY;
        new_entry->array = new_array;
        new_array->length = entry->array->length;

        for (size_t i = 0; i < new_array->length; ++i)
        {
            json_value *cloned_entry;
            json_error error = json_clone(entry->array->entry[i], &cloned_entry);
            if (error)
            {
                json_free(new_entry);
                return error;
            }
            new_array->entry[i] = cloned_entry;
        }

        *out = new_entry;
        return JSON_SUCCESS;
    }

    case JSON_OBJECT:
    {
        json_value *new_entry = malloc(sizeof(json_value));
        if (!new_entry) return JSON_ERROR_ALLOCATION;

        json_object *new_object = malloc(sizeof(json_object) + entry->object->size * sizeof(json_value*));
        if (!new_object)
        {
            free(new_entry);
            return JSON_ERROR_ALLOCATION;
        }

        if (entry->object->size)
        {
            new_object->keys = malloc(entry->object->size * sizeof(char*));
            if (!new_object->keys)
            {
                free(new_object);
                free(new_entry);
                return JSON_ERROR_ALLOCATION;
            }
        }
        else
            new_object->keys = NULL;

        new_entry->type = JSON_OBJECT;
        new_entry->object = new_object;
        new_object->size = entry->object->size;

        for (size_t i = 0; i < new_object->size; ++i)
        {
            new_object->keys[i] = strdup(entry->object->keys[i]);
            if (!new_object->keys[i])
            {
                json_free(new_entry);
                return JSON_ERROR_ALLOCATION;
            }

            json_value *cloned_entry;
            json_error error = json_clone(entry->object->entry[i], &cloned_entry);
            if (error)
            {
                json_free(new_entry);
                return error;
            }
            new_object->entry[i] = cloned_entry;
        }

        *out = new_entry;
        return JSON_SUCCESS;
    }
    }
    return JSON_ERROR_WRONG_TYPE;
}

static void free_array(json_array *array)
{
    for (size_t i = 0; i < array->length; ++i)
        json_free(array->entry[i]);
    free(array);
}

static void free_object(json_object *object)
{
    for (size_t i = 0; i < object->size; ++i)
    {
        free(object->keys[i]);
        json_free(object->entry[i]);
    }
    free(object->keys);
    free(object);
}

static void free_content(json_value *entry)
{
    switch (entry->type)
    {
    case JSON_STRING: free(entry->string);        return;
    case JSON_ARRAY:  free_array(entry->array);   return;
    case JSON_OBJECT: free_object(entry->object); return;
    default: return;
    }
}

void json_free(json_value *entry)
{
    if (!entry) return;
    free_content(entry);
    free(entry);
}

// --------------------
// JSON Getter API
// --------------------

json_error json_get_type(const json_value *entry, json_type *out)
{
    if (!entry) return JSON_ERROR_NULL;
    *out = entry->type;
    return JSON_SUCCESS;
}

json_error json_bool_get(const json_value *entry, bool *out)
{
    if (!entry || !out) return JSON_ERROR_NULL;
    CHECK_TYPE(entry, JSON_BOOL);
    *out = entry->boolean;
    return JSON_SUCCESS;
}

json_error json_number_get(const json_value *entry, double *out)
{
    if (!entry || !out) return JSON_ERROR_NULL;
    CHECK_TYPE(entry, JSON_NUMBER);
    *out = entry->number;
    return JSON_SUCCESS;
}

json_error json_string_get(const json_value *entry, const char **out)
{
    if (!entry || !out) return JSON_ERROR_NULL;
    CHECK_TYPE(entry, JSON_STRING);
    *out = entry->string;
    return JSON_SUCCESS;
}

// --------------------
// JSON Setter API
// --------------------

json_error json_set_as_null(json_value *entry)
{
    if (!entry) return JSON_ERROR_NULL;
    free_content(entry);
    *entry = (json_value) {0};
    entry->type = JSON_NULL;
    return JSON_SUCCESS;
}

json_error json_set_as_bool(json_value *entry, bool value)
{
    if (!entry) return JSON_ERROR_NULL;
    free_content(entry);
    *entry = (json_value) {0};
    entry->type = JSON_BOOL;
    entry->boolean = value;
    return JSON_SUCCESS;
}

json_error json_set_as_number(json_value *entry, double value)
{
    if (!entry) return JSON_ERROR_NULL;
    free_content(entry);
    *entry = (json_value) {0};
    entry->type = JSON_NUMBER;
    entry->number = value;
    return JSON_SUCCESS;
}

json_error json_set_as_string(json_value *entry, const char *string)
{
    if (!entry || !string) return JSON_ERROR_NULL;

    char *string_copy = strdup(string);
    if (!string_copy) return JSON_ERROR_ALLOCATION;

    free_content(entry);
    *entry = (json_value) {0};
    entry->type = JSON_STRING;
    entry->string = string_copy;
    return JSON_SUCCESS;
}

json_error json_set_as_string_nocopy(json_value *entry, char *string)
{
    if (!entry || !string) return JSON_ERROR_NULL;

    free_content(entry);
    *entry = (json_value) {0};
    entry->type = JSON_STRING;
    entry->string = string;
    return JSON_SUCCESS;
}

json_error json_set_as_array(json_value *entry)
{
    if (!entry) return JSON_ERROR_NULL;

    json_array *array = malloc(sizeof(json_array));
    if (!array) return JSON_ERROR_ALLOCATION;

    free_content(entry);
    *entry = (json_value) {0};
    entry->type = JSON_ARRAY;
    entry->array = array;
    array->length = 0;
    return JSON_SUCCESS;
}

json_error json_set_as_object(json_value *entry)
{
    if (!entry) return JSON_ERROR_NULL;

    json_object *object = malloc(sizeof(json_object));
    if (!object) return JSON_ERROR_ALLOCATION;

    free_content(entry);
    object->size = 0;
    object->keys = NULL;
    *entry = (json_value) {0};
    entry->type = JSON_OBJECT;
    entry->object = object;
    return JSON_SUCCESS;
}

// --------------------
// JSON Array API
// --------------------

json_error json_array_length(const json_value *array, size_t *out)
{
    if (!array || !out) return JSON_ERROR_NULL;
    CHECK_TYPE(array, JSON_ARRAY);

    *out = array->array->length;
    return JSON_SUCCESS;
}

json_error json_array_get(const json_value *array, size_t index, json_value **out)
{
    if (!array || !out) return JSON_ERROR_NULL;
    CHECK_TYPE(array, JSON_ARRAY);
    if (index >= array->array->length) return JSON_ERROR_INDEX_OUT_OF_BOUNDS;

    *out = array->array->entry[index];
    return JSON_SUCCESS;
}

json_error json_array_set(json_value *array, size_t index, json_value *value)
{
    if (!array || !value) return JSON_ERROR_NULL;
    CHECK_TYPE(array, JSON_ARRAY);
    if (index >= array->array->length) return JSON_ERROR_INDEX_OUT_OF_BOUNDS;

    json_value **array_slot = &array->array->entry[index];
    json_free(*array_slot);
    *array_slot = value;
    return JSON_SUCCESS;
}

json_error json_array_append(json_value *array, json_value *value)
{
    if (!array || !value) return JSON_ERROR_NULL;
    CHECK_TYPE(array, JSON_ARRAY);

    json_array *new_array = realloc(array->array, sizeof(json_array) + (array->array->length + 1) * sizeof(json_value*));
    if (!new_array) return JSON_ERROR_ALLOCATION;

    array->array = new_array;
    new_array->entry[new_array->length++] = value;
    return JSON_SUCCESS;
}

json_error json_array_insert(json_value *array, size_t index, json_value *value)
{
    if (!array || !value) return JSON_ERROR_NULL;
    CHECK_TYPE(array, JSON_ARRAY);
    if (index > array->array->length) return JSON_ERROR_INDEX_OUT_OF_BOUNDS;

    json_array *new_array = realloc(array->array, sizeof(json_array) + (array->array->length + 1) * sizeof(json_value*));
    if (!new_array) return JSON_ERROR_ALLOCATION;

    if (index < new_array->length)
        memmove(new_array->entry + index + 1, new_array->entry + index, (new_array->length - index) * sizeof(json_value*));
    new_array->entry[index] = value;
    new_array->length++;
    return JSON_SUCCESS;
}

json_error json_array_remove(json_value *array_value, size_t index, json_value **out)
{
    if (!array_value) return JSON_ERROR_NULL;
    CHECK_TYPE(array_value, JSON_ARRAY);
    if (index >= array_value->array->length) return JSON_ERROR_INDEX_OUT_OF_BOUNDS;

    json_value *removed = array_value->array->entry[index];
    if (index < array_value->array->length - 1)
        memmove(array_value->array->entry + index, array_value->array->entry + index + 1, (array_value->array->length - index - 1) * sizeof(json_value*));

    array_value->array->length--;

    if (out)
        *out = removed;
    else
        json_free(removed);

    return JSON_SUCCESS;
}

// --------------------
// JSON Object API
// --------------------

json_error json_object_size(const json_value *object, size_t *out)
{
    if (!object || !out) return JSON_ERROR_NULL;
    CHECK_TYPE(object, JSON_OBJECT);

    *out = object->object->size;
    return JSON_SUCCESS;
}

json_error json_object_has_key(const json_value *object, const char *key, bool *out)
{
    if (!object || !key || !out) return JSON_ERROR_NULL;
    CHECK_TYPE(object, JSON_OBJECT);

    for (size_t i = 0; i < object->object->size; ++i)
    {
        if (!strcmp(object->object->keys[i], key))
        {
            *out = true;
            return JSON_SUCCESS;
        }
    }

    *out = false;
    return JSON_SUCCESS;
}

json_error json_object_get(const json_value *object, const char *key, json_value **out)
{
    if (!object || !key || !out) return JSON_ERROR_NULL;
    CHECK_TYPE(object, JSON_OBJECT);

    for (size_t i = 0; i < object->object->size; ++i)
    {
        if (!strcmp(object->object->keys[i], key))
        {
            *out = object->object->entry[i];
            return JSON_SUCCESS;
        }
    }

    return JSON_ERROR_KEY_NOT_FOUND;
}

json_error json_object_set(json_value *object, const char *key, json_value *value)
{
    if (!object || !key || !value) return JSON_ERROR_NULL;
    CHECK_TYPE(object, JSON_OBJECT);

    for (size_t i = 0; i < object->object->size; ++i)
    {
        if (!strcmp(object->object->keys[i], key))
        {
            json_free(object->object->entry[i]);
            object->object->entry[i] = value;
            return JSON_SUCCESS;
        }
    }

    char *key_copy = strdup(key);
    if (!key_copy) return JSON_ERROR_ALLOCATION;

    json_object *new_object = realloc(object->object, sizeof(json_object) + (object->object->size + 1) * sizeof(json_value*));
    if (!new_object)
    {
        free(key_copy);
        return JSON_ERROR_ALLOCATION;
    }
    object->object = new_object;

    char **new_keys = realloc(new_object->keys, (object->object->size + 1) * sizeof(char*));
    if (!new_keys)
    {
        free(key_copy);
        return JSON_ERROR_ALLOCATION;
    }

    new_keys[new_object->size] = key_copy;
    new_object->keys = new_keys;
    new_object->entry[object->object->size] = value;
    new_object->size++;
    return JSON_SUCCESS;
}

json_error json_object_remove(json_value *object, const char *key, json_value **out)
{
    if (!object || !key) return JSON_ERROR_NULL;
    CHECK_TYPE(object, JSON_OBJECT);

    for (size_t i = 0; i < object->object->size; ++i)
    {
        char *old_key = object->object->keys[i];
        if (strcmp(old_key, key))
            continue;
        free(old_key);
        
        json_value *removed = object->object->entry[i];
        if (i < object->object->size - 1)
        {
            memmove(object->object->keys + i, object->object->keys + i + 1, (object->object->size - i - 1) * sizeof(char*));
            memmove(object->object->entry + i, object->object->entry + i + 1, (object->object->size - i - 1) * sizeof(json_value*));
        }

        object->object->size--;

        if (out)
            *out = removed;
        else
            json_free(removed);

        return JSON_SUCCESS;
    }

    return JSON_ERROR_KEY_NOT_FOUND;
}

// ----------
// JSON Parsing
// ----------

const json_parse_options JSON_DEFAULT_PARSE_OPTIONS = {
    .error_info = NULL,
    .max_depth = DEFAULT_MAX_DEPTH
};

typedef struct json_parser {
    const json_parse_options *options;
    json_error_info *error_info;

    union
    {
        FILE *input_file;
        const char *input_string;
    };
    int (*getc)(struct json_parser *parser);
    
    size_t line, column;
    size_t depth;
    int last_c;
    json_error error;
} json_parser;

// Reports a parsing error at the current location.
static void report_parsing_error(json_parser *parser, json_error error_type, const char *error_fmt, ...)
{
    parser->error = error_type;

    if (!parser->error_info) return;
    parser->error_info->line = parser->line;
    parser->error_info->column = parser->column;
    parser->error_info->error = error_type;

    va_list args, args_copy;
    va_start(args, error_fmt);

    va_copy(args_copy, args);
    int size = vsnprintf(NULL, 0, error_fmt, args_copy);
    va_end(args_copy);
    if (size < 0)
    {
        va_end(args);
        strcpy(parser->error_info->message, "invalid error message format");
        return;
    }

    vsnprintf(parser->error_info->message, 256, error_fmt, args);
    va_end(args);
}

static bool parse_entry(json_parser *parser, json_value **out);

// Reads the next character and updates line/column.
static void consume(json_parser *parser)
{
    if (parser->last_c == EOF) return;
    parser->last_c = parser->getc(parser);

    if (parser->last_c == '\n')
    {
        parser->line++;
        parser->column = 0;
    }
    else
        parser->column++;
}

// Skips whitespace characters.
static void skip_blank(json_parser *parser)
{
    while (isspace(parser->last_c) && parser->last_c != EOF)
        consume(parser);
}

// Verifies that the next character is as expected. If not, reports error.
static bool expect(json_parser *parser, char expected_c)
{
    if (parser->last_c != expected_c)
    {
        report_parsing_error(parser, JSON_ERROR_UNEXPECTED_CHARACTER,
            "expected '%c', found '%c'", expected_c, parser->last_c);
        return true;
    }

    consume(parser);
    skip_blank(parser);
    return false;
}

// Builds and returns a string from characters matching a predicate.
static bool get_string(json_parser *parser, bool (*predicate)(int c), char **out)
{
    string_builder builder = {0};

    while (predicate(parser->last_c))
    {
        if (string_builder_append(&builder, parser->last_c))
            goto alloc_error;
        consume(parser);
    }

    skip_blank(parser);

    char *string;
    if (string_builder_build(&builder, &string))
        goto alloc_error;

    *out = string;
    return false;

alloc_error:
    string_builder_free(&builder);
    report_parsing_error(parser, JSON_ERROR_ALLOCATION, "couldn't reallocate string buffer");
    return true;
}

// Converts a hexadecimal digit to its value. Returns -1 if invalid.
static int hex_digit_to_value(char c)
{
    if ('0' <= c && c <= '9') return c - '0';
    if ('a' <= c && c <= 'f') return c - 'a' + 10;
    if ('A' <= c && c <= 'F') return c - 'A' + 10;
    return -1;
}

// Parses 4 hexadecimal digits into a UTF-16 code unit.
static bool parse_utf16_code_unit(json_parser *parser, uint32_t *out)
{
    uint32_t code_point = 0;
    for (size_t i = 0; i < 4; ++i)
    {
        int digit = hex_digit_to_value(parser->last_c);
        if (digit < 0)
        {
            report_parsing_error(parser, JSON_ERROR_UNICODE, "invalid hexadecimal digit '%c'", parser->last_c);
            return true;
        }
        code_point = (code_point << 4) | digit;
        consume(parser);
    }
    *out = code_point;
    return false;
}

// Parses Unicode escape sequences, including surrogate pairs.
static bool parse_utf16_escape(json_parser *parser, uint32_t *out)
{
    uint32_t code_point;
    if (parse_utf16_code_unit(parser, &code_point)) return true;

    if (code_point < 0xD800 || 0xDFFF < code_point)
    {
        *out = code_point;
        return false;
    }

    if (0xDBFF < code_point)
    {
        report_parsing_error(parser, JSON_ERROR_UNICODE, "invalid high surrogate range U+%04X", code_point);
        return true;
    }

    // Comma operators are evil, but this is the most readable way to do this.
    if (parser->last_c != '\\' ||
        (consume(parser), parser->last_c) != 'u')
    {
        report_parsing_error(parser, JSON_ERROR_UNICODE, "missing low surrogate after high surrogate U+%04X", code_point);
        return true;
    }
    consume(parser);

    uint32_t low_surrogate;
    if (parse_utf16_code_unit(parser, &low_surrogate))
        return true;

    if (low_surrogate < 0xDC00 || 0xDFFF < low_surrogate)
    {
        report_parsing_error(parser, JSON_ERROR_UNICODE, "invalid low surrogate range U+%04X", low_surrogate);
        return true;
    }  

    code_point = 0x10000 + (((code_point & 0x3FF) << 10) + (low_surrogate & 0x3FF));
    if (code_point > 0x10FFFF)
    {
        report_parsing_error(parser, JSON_ERROR_UNICODE, "invalid code point U+%04X", code_point);
        return true;
    }

    *out = code_point;
    return false;
}

// Processes current escape sequence.
static bool consume_escaped_character(json_parser *parser, string_builder *builder)
{
    if (expect(parser, '\\'))
        return true;

    char escape_sequence = parser->last_c;
    consume(parser);

    switch (escape_sequence)
    {
    case '"':  return string_builder_append(builder, '"');  break;
    case '\\': return string_builder_append(builder, '\\'); break;
    case '/':  return string_builder_append(builder, '/');  break;
    case 'b':  return string_builder_append(builder, '\b'); break;
    case 'f':  return string_builder_append(builder, '\f'); break;
    case 'n':  return string_builder_append(builder, '\n'); break;
    case 'r':  return string_builder_append(builder, '\r'); break;
    case 't':  return string_builder_append(builder, '\t'); break;

    case 'u':
        {
        uint32_t code_point;
        if (parse_utf16_escape(parser, &code_point)) return true;
        return string_builder_append_utf_code_point(builder, code_point);
        }

    default:
        report_parsing_error(parser, JSON_ERROR_ESCAPE_SEQUENCE, "invalid escape sequence '%c'", parser->last_c);
        return true;
    }
}

// Parses and returns a JSON quoted string.
static bool get_quoted_string(json_parser *parser, char **out)
{
    if (expect(parser, '"'))
        return true;

    string_builder builder = {0};

    while (parser->last_c != '"'
        && parser->last_c != EOF
        && parser->last_c != '\n')
    {
        if (parser->last_c == '\\')
        {
            if (consume_escaped_character(parser, &builder))
                goto clean_up;
            continue;
        }

        if (string_builder_append(&builder, parser->last_c))
            goto alloc_error;
        consume(parser);
    }

    if (expect(parser, '"'))
        goto clean_up;
    
    if (string_builder_build(&builder, out))
        goto alloc_error;
    return false;

alloc_error:
    report_parsing_error(parser, JSON_ERROR_ALLOCATION, "couldn't reallocate string buffer");
clean_up:
    string_builder_free(&builder);
    return true;
}

// Creates a JSON entry from a string.
static bool parse_string(json_parser *parser, json_value **out)
{
    char *string;
    if (get_quoted_string(parser, &string)) return true;
    return json_string_create_nocopy(string, out);
}

// Checks if character can be part of a JSON identifier.
static bool is_part_of_identifier(int c)
{
    return isalpha(c);
}

// Parses JSON identifiers (null, true, false).
static bool parse_identifier(json_parser *parser, json_value **out)
{
    char *buffer;
    if (get_string(parser, is_part_of_identifier, &buffer))
        return true;

    json_error error;
    if (!strcmp(buffer, "null"))
        error = json_null_create(out);
    else if (!strcmp(buffer, "true"))
        error = json_bool_create(true, out);
    else if (!strcmp(buffer, "false"))
        error = json_bool_create(false, out);
    else
    {
        free(buffer);
        report_parsing_error(parser, JSON_ERROR_UNEXPECTED_IDENTIFIER, "unknown identifier '%s'", buffer);
        return true;
    }
    
    free(buffer);
    if (error != JSON_SUCCESS)
    {
        report_parsing_error(parser, error, "failed to create identifier '%s' entry", buffer);
        return true;
    }

    return false;
}

// Checks if character can be part of a number.
static bool is_part_of_number(int c)
{
    return isdigit(c) || c == '+' || c == '-' || c == '.' || c == 'e' || c == 'E';
}

// Converts a string of digits (and other characters) into a JSON number entry.
static bool parse_number(json_parser *parser, json_value **out)
{
    char *buffer;
    if (get_string(parser, is_part_of_number, &buffer))
        return true;

    char *number_end;
    double number = strtod(buffer, &number_end);
    char final_char = *number_end;
    if (final_char != '\0')
    {
        report_parsing_error(parser, JSON_ERROR_NUMBER_FORMAT, "invalid number format '%s'", buffer);
        goto clean_up;
    }
    if (errno == ERANGE)
    {
        report_parsing_error(parser, JSON_ERROR_NUMBER_FORMAT, "number '%s' out of range", buffer);
        goto clean_up;
    }
    free(buffer);

    return json_number_create(number, out);

clean_up:
    free(buffer);
    return true;
}

// Parses a JSON array.
static bool parse_array(json_parser *parser, json_value **out)
{
    if (expect(parser, '['))
        return true;

    json_value *array_value;
    json_error error = json_array_create(&array_value);
    if (error)
    {
        report_parsing_error(parser, error, "failed to create array entry");
        return true;
    }

    bool first_entry = true;
    while (parser->last_c != ']' && parser->last_c != EOF)
    {
        if (!first_entry && expect(parser, ','))
            goto clean_up;
        first_entry = false;

        json_value *this_entry;
        if (parse_entry(parser, &this_entry))
            goto clean_up;

        json_array_append(array_value, this_entry);
    }
    
    if (expect(parser, ']'))
        goto clean_up;

    *out = array_value;
    return false;

clean_up:
    json_free(array_value);
    return true;
}

// Parses a JSON object.
static bool parse_object(json_parser *parser, json_value **out)
{
    if (expect(parser, '{'))
        return true;

    json_value *object_value;
    json_error error = json_object_create(&object_value);
    if (error)
    {
        report_parsing_error(parser, error, "failed to create object entry");
        return true;
    }

    bool first_entry = true;
    while (parser->last_c != '}' && parser->last_c != EOF)
    {
        if (!first_entry && expect(parser, ','))
            goto clean_up;
        first_entry = false;

        char *key_string;
        if (get_quoted_string(parser, &key_string))
            goto clean_up;

        if (expect(parser, ':'))
        {
            free(key_string);
            goto clean_up;
        }

        json_value *this_entry;
        if (parse_entry(parser, &this_entry))
        {
            free(key_string);
            goto clean_up;
        }

        json_object_set(object_value, key_string, this_entry);
    }

    if (expect(parser, '}'))
        goto clean_up;

    *out = object_value;
    return false;

clean_up:
    json_free(object_value);
    return true;
}

// Determines the JSON type and parse the corresponding entry.
static bool parse_entry(json_parser *parser, json_value **out)
{
    parser->depth++;
    if (parser->depth > parser->options->max_depth)
    {
        report_parsing_error(parser, JSON_ERROR_MAX_DEPTH, "maximum depth (%zu) exceeded", parser->options->max_depth);
        return true;
    }

    bool error = true;
    if (parser->last_c == '[')
        error = parse_array(parser, out);
    else if (parser->last_c == '{')
        error = parse_object(parser, out);
    else if (parser->last_c == '"')
        error = parse_string(parser, out);
    else if (isalpha(parser->last_c))
        error = parse_identifier(parser, out);
    else if (isdigit(parser->last_c) || parser->last_c == '-')
        error = parse_number(parser, out);
    else
        report_parsing_error(parser, JSON_ERROR_UNEXPECTED_CHARACTER, "unexpected character '%c'", parser->last_c);

    parser->depth--;

    return error;
}

// Main entry point for parsing a JSON input.
static json_error parse(json_parser *parser, json_value **out)
{
    parser->line = 1;
    parser->column = 0;
    parser->depth = 0;
    parser->error = JSON_SUCCESS;

    consume(parser);
    skip_blank(parser);

    if (parser->last_c == EOF)
    {
        *out = NULL;
        return JSON_SUCCESS;
    }

    json_value *root = NULL;
    if (parse_entry(parser, &root))
        goto clean_up;

    if (parser->last_c != EOF)
    {
        report_parsing_error(parser, JSON_ERROR_UNEXPECTED_CHARACTER,
            "expected end of file, found '%c'", parser->last_c);
        goto clean_up;
    }

    *out = root;
    return JSON_SUCCESS;

clean_up:
    if (root)
        json_free(root);
    return parser->error;
}

// --------------------
// Input Source Functions
// --------------------

static int getc_from_string(json_parser *parser)
{
    const char *string = parser->input_string;
    if (*string == '\0')
        return EOF;
    parser->input_string = string + 1;
    return *string;
}

json_error json_parse_string(const char *string, json_value **value, const json_parse_options *options)
{
    if (!value) return JSON_ERROR_NULL;
    if (!options) options = &JSON_DEFAULT_PARSE_OPTIONS;

    json_parser parser = {
        .input_string = string,
        .getc = getc_from_string,
        .options = options
    };

    if (!string)
    {
        report_parsing_error(&parser, JSON_ERROR_NULL, "string is NULL");
        return JSON_ERROR_NULL;
    }

    return parse(&parser, value);
}


static int getc_from_file(json_parser *parser)
{
    return fgetc(parser->input_file);
}

json_error json_parse_file(FILE *file, json_value **value, const json_parse_options *options)
{
    if (!value) return JSON_ERROR_NULL;
    if (!options) options = &JSON_DEFAULT_PARSE_OPTIONS;
    
    json_parser parser = {
        .input_file = file,
        .getc = getc_from_file,
        .options = options
    };

    if (!file)
    {
        report_parsing_error(&parser, JSON_ERROR_NULL, "file is NULL");
        return JSON_ERROR_NULL;
    }

    return parse(&parser, value);
}

// --------------------
// JSON Printing Functions
// --------------------

const json_format_options JSON_DEFAULT_FORMAT_OPTIONS = {
    .error_info = NULL,
    .indent_size = 2,
    .max_depth = DEFAULT_MAX_DEPTH
};

typedef struct json_serializer {
    const json_format_options *options;
    json_error_info *error_info;

    union
    {
        FILE *output_file;
        string_builder *output_builder;
    };
    void (*putc)(struct json_serializer *serializer, char c);
    void (*puts)(struct json_serializer *serializer, const char *str);
    void (*printf)(struct json_serializer *serializer, const char *format, ...);

    size_t depth;
    json_error error;
} json_serializer;

static void serialize_value(json_serializer *serializer, const json_value *entry);

static void report_serialization_error(json_serializer *serializer, json_error error_type, const char *error_fmt, ...)
{
    serializer->error = error_type;

    if (!serializer->error_info) return;
    serializer->error_info->error = error_type;

    va_list args, args_copy;
    va_start(args, error_fmt);

    va_copy(args_copy, args);
    int size = vsnprintf(NULL, 0, error_fmt, args_copy);
    va_end(args_copy);
    if (size < 0)
    {
        va_end(args);
        strcpy(serializer->error_info->message, "invalid error message format");
        return;
    }

    vsnprintf(serializer->error_info->message, 256, error_fmt, args);
    va_end(args);
}

// Writes indentation spaces to the file.
static void serialize_indent(json_serializer *serializer)
{
    size_t indent = serializer->options->indent_size;
    if (indent == 0) return;

    serializer->putc(serializer, '\n');
    for (indent *= serializer->depth; indent != 0; --indent)
        serializer->putc(serializer, ' ');
}

// Prints a JSON array with proper formatting.
static void serialize_array(json_serializer *serializer, const json_array *array)
{
    serializer->putc(serializer, '[');

    serializer->depth++;
    const json_array *entry = array;
    for (size_t i = 0; i < array->length; ++i)
    {
        if (i > 0) serializer->putc(serializer, ',');
        serialize_indent(serializer);
        serialize_value(serializer, entry->entry[i]);
    }
    serializer->depth--;

    serialize_indent(serializer);
    serializer->putc(serializer, ']');
}

// Writes a JSON string escaping special characters.
static void serialize_escape_string(json_serializer *serializer, char *str)
{
    for (; *str != '\0'; ++str)
    {
        if ((unsigned char)*str < 0x20 || *str == 0x7F)
        {
            serializer->printf(serializer, "\\u00%02X", *str & 0xFF);
            continue;
        }

        switch (*str)
        {
        case '"':  serializer->puts(serializer, "\\\""); break;
        case '\\': serializer->puts(serializer, "\\\\"); break;
        case '/':  serializer->puts(serializer, "\\/"); break;
        case '\b': serializer->puts(serializer, "\\b"); break;
        case '\f': serializer->puts(serializer, "\\f"); break;
        case '\n': serializer->puts(serializer, "\\n"); break;
        case '\r': serializer->puts(serializer, "\\r"); break;
        case '\t': serializer->puts(serializer, "\\t"); break;
        default:   serializer->putc(serializer, *str); break;
        }
    }
}

// Prints a JSON object with proper formatting.
static void serialize_object(json_serializer *serializer, const json_object *object)
{
    serializer->putc(serializer, '{');

    serializer->depth++;
    const json_object *entry = object;
    bool is_compact = serializer->options->indent_size == 0;

    for (size_t i = 0; i < object->size; ++i)
    {
        if (i > 0) serializer->putc(serializer, ',');
        serialize_indent(serializer);
        serializer->putc(serializer, '"');
        serialize_escape_string(serializer, entry->keys[i]);
        serializer->puts(serializer, is_compact ? "\":" : "\": ");
        serialize_value(serializer, entry->entry[i]);
    }

    serializer->depth--;

    serialize_indent(serializer);
    serializer->putc(serializer, '}');
}

static void serialize_value(json_serializer *serializer, const json_value *entry)
{
    if (!entry) return;

    if (serializer->depth > serializer->options->max_depth)
    {
        serializer->puts(serializer, "null");
        report_serialization_error(serializer, JSON_ERROR_MAX_DEPTH, "maximum depth exceeded");
        return;
    }

    switch (entry->type)
    {
    case JSON_NULL:
        serializer->puts(serializer, "null");
        break;

    case JSON_NUMBER:
        serializer->printf(serializer, "%g", entry->number);
        break;

    case JSON_STRING:
        serializer->putc(serializer, '"');
        serialize_escape_string(serializer, entry->string);
        serializer->putc(serializer, '"');
        break;

    case JSON_BOOL:
        serializer->puts(serializer, entry->boolean ? "true" : "false");
        break;

    case JSON_ARRAY:
        serialize_array(serializer, entry->array);
        break;

    case JSON_OBJECT:
        serialize_object(serializer, entry->object);
        break;
    }
}

static void putc_to_file(json_serializer *serializer, char c)
{
    fputc(c, serializer->output_file);
}

static void puts_to_file(json_serializer *serializer, const char *str)
{
    fputs(str, serializer->output_file);
}

static void printf_to_file(json_serializer *serializer, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vfprintf(serializer->output_file, format, args);
    va_end(args);
}

json_error serialize(json_serializer *serializer, const json_value *entry)
{
    serializer->depth = 0;
    serializer->error = JSON_SUCCESS;
    serialize_value(serializer, entry);
    if (serializer->options->indent_size != 0)
        serializer->putc(serializer, '\n');
    return serializer->error;
}

json_error json_serialize_to_file(const json_value *entry, FILE *file, const json_format_options *options)
{
    if (!options) options = &JSON_DEFAULT_FORMAT_OPTIONS;

    json_serializer serializer = {
        .options = options,
        .putc = putc_to_file,
        .puts = puts_to_file,
        .printf = printf_to_file,
        .output_file = file
    };

    if (!file)
    {
        report_serialization_error(&serializer, JSON_ERROR_NULL, "file is NULL");
        return JSON_ERROR_NULL;
    }

    return serialize(&serializer, entry);
}

static void putc_to_string(json_serializer *serializer, char c)
{
    string_builder_append(serializer->output_builder, c);
}

static void puts_to_string(json_serializer *serializer, const char *str)
{
    string_builder_append_string(serializer->output_builder, str);
}

static void printf_to_string(json_serializer *serializer, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    string_builder_append_format(serializer->output_builder, format, args);
    va_end(args);
}

json_error json_serialize_to_string(const json_value *entry, char **dst, const json_format_options *options)
{
    if (!dst) return JSON_ERROR_NULL;
    if (!options) options = &JSON_DEFAULT_FORMAT_OPTIONS;

    string_builder builder = {0};
    json_serializer serializer = {
        .options = options,
        .putc = putc_to_string,
        .puts = puts_to_string,
        .printf = printf_to_string,
        .output_builder = &builder
    };

    if (serialize(&serializer, entry) != JSON_SUCCESS)
        goto clean_up;

    if (string_builder_build(&builder, dst))
    {
        report_serialization_error(&serializer, JSON_ERROR_ALLOCATION, "couldn't reallocate string buffer");
        goto clean_up;
    }
    return JSON_SUCCESS;

clean_up:
    string_builder_free(&builder);
    return serializer.error;
}
