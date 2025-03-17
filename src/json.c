#include "json.h"

#include <stdint.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdarg.h>

#include "constants.h"

#define INITIAL_BUFFER_SIZE 16

static char *strdup(const char *string)
{
    size_t size = strlen(string);
    char *string_copy = malloc(size + 1);
    if (!string_copy)
        return NULL;
    strncpy(string_copy, string, size + 1);
    return string_copy;
}

typedef struct string_builder {
    size_t allocated_size;
    size_t size;
    char *data;
} string_builder;

static void string_builder_free(string_builder *builder)
{
    free(builder->data);
    builder->allocated_size = 0;
    builder->size = 0;
    builder->data = NULL;
}

static bool string_builder_ensure_capacity(string_builder *builder, size_t min_capacity)
{
    if (min_capacity + 1 > builder->allocated_size)
    {
        char *new_data;
        if (builder->allocated_size == 0)
        {
            builder->allocated_size = INITIAL_BUFFER_SIZE;
            new_data = malloc(builder->allocated_size);
        }
        else
        {
            builder->allocated_size *= 2;
            new_data = realloc(builder->data, builder->allocated_size);
        }
        if (!new_data)
            return false;
        builder->data = new_data;
    }
    return true;
}

static bool string_builder_append(string_builder *builder, char c)
{
    bool has_capacity = string_builder_ensure_capacity(builder, builder->size + 1);
    if (!has_capacity)
        return false;

    builder->data[builder->size++] = c;
    builder->data[builder->size] = '\0';
    return true;
}

static bool string_builder_append_utf_code_point(string_builder *builder, uint32_t code_point)
{
    bool has_capacity = string_builder_ensure_capacity(builder, builder->size + 4);
    if (!has_capacity)
        return false;

    if (code_point <= 0x7F)
        builder->data[builder->size++] = code_point;
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
    else
    {
        builder->data[builder->size++] = 0xF0 | ((code_point >> 18) & 0x07);
        builder->data[builder->size++] = 0x80 | ((code_point >> 12) & 0x3F);
        builder->data[builder->size++] = 0x80 | ((code_point >> 6) & 0x3F);
        builder->data[builder->size++] = 0x80 | (code_point & 0x3F);
    }
    builder->data[builder->size] = '\0';

    return true;
}

static char* string_builder_build(string_builder *builder)
{
    char *string = realloc(builder->data, builder->size + 1);
    if (!string)
        return NULL;
    string[builder->size] = '\0';
    builder->allocated_size = 0;
    builder->size = 0;
    builder->data = NULL;
    return string;
}


typedef struct json_array {
    struct json_array *next;
    json_entry *entry;
} json_array;

typedef struct json_object {
    struct json_object *next;
    char *key;
    json_entry *entry;
} json_object;

typedef struct json_entry {
    json_type type;
    union {
        double number;
        char *string;
        bool boolean;
        json_array *array;
        json_object *object;
    };
} json_entry;

typedef struct json_parser {
    FILE *file;
    int last_c;
    size_t line, column;
    bool error;
} json_parser;

static json_entry* parse_entry(json_parser *parser);
static void fprint_entry(FILE *file, const json_entry *entry, size_t depth);

static void print_error(json_parser *parser, const char *error_fmt, ...)
{
    va_list args;
    va_start(args, error_fmt);

    fprintf(stderr, PROGRAM_NAME": error:%zu:%zu: ", parser->line, parser->column);
    vfprintf(stderr, error_fmt, args);
    fputc('\n', stderr);

    va_end(args);
    parser->error = true;
}

static void free_array(json_array *array)
{
    while (array)
    {
        json_array *next = array->next;
        json_free(array->entry);
        free(array);
        array = next;
    }
}

static void free_object(json_object *object)
{
    while (object)
    {
        json_object *next = object->next;
        json_free(object->entry);
        free(object->key);
        free(object);
        object = next;
    }
}

void json_free(json_entry *entry)
{
    if (!entry)
        return;

    switch (entry->type)
    {
    case JSON_STRING:
        free(entry->string);
        break;

    case JSON_ARRAY:
        free_array(entry->array);
        break;

    case JSON_OBJECT:
        free_object(entry->object);
        break;
    
    default:
        break;
    }
    free(entry);
}

static void consume(json_parser *parser)
{
    if (parser->last_c == EOF)
        return;

    parser->last_c = fgetc(parser->file);
    if (parser->last_c == '\n')
    {
        parser->line++;
        parser->column = 0;
    }
    else
        parser->column++;
}

static void skip_blank(json_parser *parser)
{
    while (isspace(parser->last_c) && parser->last_c != EOF)
        consume(parser);
}

static bool expect(json_parser *parser, char expected_c)
{
    if (parser->last_c != expected_c)
    {
        print_error(parser, "expected '%c', found '%c'", expected_c, parser->last_c);
        return false;
    }
    consume(parser);
    skip_blank(parser);
    return true;
}

static char* get_string(json_parser *parser, bool (*predicate)(int c))
{
    string_builder builder = {0};

    while (predicate(parser->last_c))
    {
        bool success = string_builder_append(&builder, parser->last_c);
        if (!success)
        {
            string_builder_free(&builder);
            print_error(parser, "couldn't reallocate buffer");
            return NULL;
        }
        consume(parser);
    }

    skip_blank(parser);

    char *string = string_builder_build(&builder);
    if (!string)
    {
        string_builder_free(&builder);
        print_error(parser, "couldn't reallocate buffer");
        return NULL;
    }

    return string;
}

static int hex_digit_to_dec(char c)
{
    if ('0' <= c && c <= '9') return c - '0';
    if ('a' <= c && c <= 'f') return c - 'a' + 10;
    if ('A' <= c && c <= 'F') return c - 'A' + 10;
    return -1;
}

static uint32_t parse_utf16_code_unit(json_parser *parser)
{
    uint32_t code_point = 0;
    for (size_t i = 0; i < 4; ++i)
    {
        int digit = hex_digit_to_dec(parser->last_c);
        if (digit < 0)
        {
            print_error(parser, "invalid hexadecimal digit '%c'", parser->last_c);
            return 0;
        }
        code_point = (code_point << 4) | digit;
        consume(parser);
    }
    return code_point;
}

static uint32_t parse_utf16_escape(json_parser *parser)
{
    uint32_t code_point = parse_utf16_code_unit(parser);
    if (parser->error)
        return 0;

    if (code_point < 0xD800 || 0xDFFF < code_point)
        return code_point;

    if (code_point < 0xD800 || 0xDBFF < code_point)
    {
        print_error(parser, "invalid high surrogate range U+%04X", code_point);
        return 0;
    }

    expect(parser, '\\');
    if (parser->error)
        return 0;
    expect(parser, 'u');
    if (parser->error)
        return 0;

    uint32_t low_surrogate = parse_utf16_code_unit(parser);
    if (parser->error)
        return 0;

    if (low_surrogate < 0xDC00 || 0xDFFF < low_surrogate)
    {
        print_error(parser, "invalid low surrogate range U+%04X", low_surrogate);
        return 0;
    }  

    return 0x10000 + (((code_point & 0x3FF) << 10) + (low_surrogate & 0x3FF));
}

static bool consume_escaped_character(json_parser *parser, string_builder *builder)
{
    assert(parser->last_c == '\\');

    consume(parser);
    bool success;
    switch (parser->last_c)
    {
    case '"':  success = string_builder_append(builder, '"');  break;
    case '\\': success = string_builder_append(builder, '\\'); break;
    case '/':  success = string_builder_append(builder, '/');  break;
    case 'b':  success = string_builder_append(builder, '\b'); break;
    case 'f':  success = string_builder_append(builder, '\f'); break;
    case 'n':  success = string_builder_append(builder, '\n'); break;
    case 'r':  success = string_builder_append(builder, '\r'); break;
    case 't':  success = string_builder_append(builder, '\t'); break;

    case 'u':
        consume(parser);
        uint32_t code_point = parse_utf16_escape(parser);
        if (parser->error)
            return false;
        return string_builder_append_utf_code_point(builder, code_point);

    default:
        print_error(parser, "invalid escape sequence '%c'", parser->last_c);
        return false;
    }
    consume(parser);
    return success;
}

static char* get_quoted_string(json_parser *parser)
{
    assert(parser->last_c == '"');

    consume(parser);

    string_builder builder = {0};

    while (parser->last_c != '"'
        && parser->last_c != EOF
        && parser->last_c != '\n')
    {
        if (parser->last_c == '\\')
        {
            consume_escaped_character(parser, &builder);
            if (parser->error)
            {
                string_builder_free(&builder);
                return NULL;
            }
            continue;
        }
        else
        {
            bool success = string_builder_append(&builder, parser->last_c);
            if (!success)
            {
                string_builder_free(&builder);
                print_error(parser, "couldn't reallocate buffer");
                return NULL;
            }
            consume(parser);
        }
    }

    if (!expect(parser, '"'))
    {
        string_builder_free(&builder);
        return NULL;
    }
    
    char *string = string_builder_build(&builder);
    if (!string)
    {
        string_builder_free(&builder);
        print_error(parser, "couldn't reallocate buffer");
        return NULL;
    }

    return string;
}

static json_entry* parse_string(json_parser *parser)
{
    assert(parser->last_c == '"');

    char *string = get_quoted_string(parser);
    if (parser->error)
        return NULL;

    json_entry *entry = malloc(sizeof(json_entry));
    entry->type = JSON_STRING;
    entry->string = string;
    return entry;
}

static bool is_part_of_identifier(int c)
{
    return isalpha(c);
}

static json_entry* parse_identifier(json_parser *parser)
{
    assert(isalpha(parser->last_c));

    char *buffer = get_string(parser, is_part_of_identifier);
    if (parser->error)
        return NULL;

    if (!strcmp(buffer, "null"))
    {
        free(buffer);
        json_entry *entry = malloc(sizeof(json_entry));
        *entry = (json_entry) {0};
        return entry;
    }
    if (!strcmp(buffer, "true"))
    {
        free(buffer);
        json_entry *entry = malloc(sizeof(json_entry));
        entry->type = JSON_BOOL;
        entry->boolean = true;
        return entry;
    }
    if (!strcmp(buffer, "false"))
    {
        free(buffer);
        json_entry *entry = malloc(sizeof(json_entry));
        entry->type = JSON_BOOL;
        entry->boolean = false;
        return entry;
    }

    print_error(parser, "unknown identifier '%s'", buffer);
    return NULL;
}

static bool is_part_of_number(int c)
{
    return isdigit(c) || c == '+' || c == '-' || c == '.' || c == 'e' || c == 'E';
}

static json_entry* parse_number(json_parser *parser)
{
    char *buffer = get_string(parser, is_part_of_number);
    if (parser->error)
        return NULL;

    char *final_char;
    double number = strtod(buffer, &final_char);
    if (*final_char != '\0' || errno == ERANGE)
    {
        free(buffer);
        print_error(parser, "invalid number '%s'", buffer);
        return NULL;
    }
    free(buffer);

    json_entry *entry = malloc(sizeof(json_entry));
    entry->type = JSON_NUMBER;
    entry->number = number;
    return entry;
}

static json_entry* parse_array(json_parser *parser)
{
    assert(parser->last_c == '[');

    consume(parser);
    skip_blank(parser);

    json_array *array_head = NULL;
    json_array *array_tail = NULL;
    while (parser->last_c != ']' && parser->last_c != EOF)
    {
        if (array_head &&
            !expect(parser, ','))
        {
            free_array(array_head);
            return NULL;
        }

        json_entry *this_entry = parse_entry(parser);
        if (parser->error)
        {
            free_array(array_head);
            return NULL;
        }

        json_array *new_array = malloc(sizeof(json_array));
        new_array->entry = this_entry;
        new_array->next = NULL;
        
        if (!array_head)
            array_head = new_array;
        else
            array_tail->next = new_array;
        array_tail = new_array;
    }
    
    if (!expect(parser, ']'))
    {
        free_array(array_head);
        return NULL;
    }

    json_entry *new_entry = malloc(sizeof(json_entry));
    new_entry->type = JSON_ARRAY;
    new_entry->array = array_head;
    return new_entry;
}

static json_entry* parse_object(json_parser *parser)
{
    assert(parser->last_c == '{');

    consume(parser);
    skip_blank(parser);

    json_object *object_head = NULL;
    json_object *object_tail = NULL;
    while (parser->last_c != '}' && parser->last_c != EOF)
    {
        if (object_head &&
            !expect(parser, ','))
        {
            free_object(object_head);
            return NULL;
        }

        char *key_string = get_quoted_string(parser);
        if (parser->error)
        {
            free_object(object_head);
            return NULL;
        }

        if (!expect(parser, ':'))
        {
            free_object(object_head);
            return NULL;
        }

        json_entry *this_entry = parse_entry(parser);
        if (!this_entry)
        {
            free_object(object_head);
            return NULL;
        }

        json_object *new_object = malloc(sizeof(json_object));
        new_object->key = key_string;
        new_object->entry = this_entry;
        new_object->next = NULL;

        if (!object_head)
            object_head = new_object;
        else
            object_tail->next = new_object;
        object_tail = new_object;
    }

    if (!expect(parser, '}'))
    {
        free_object(object_head);
        return NULL;
    }

    json_entry *new_entry = malloc(sizeof(json_entry));
    new_entry->type = JSON_OBJECT;
    new_entry->object = object_head;
    return new_entry;
}

static json_entry* parse_entry(json_parser *parser)
{
    skip_blank(parser);
    if (parser->last_c == '[')
        return parse_array(parser);
    else if (parser->last_c == '{')
        return parse_object(parser);
    else if (parser->last_c == '"')
        return parse_string(parser);
    else if (isalpha(parser->last_c))
        return parse_identifier(parser);
    else if (isdigit(parser->last_c) || parser->last_c == '-')
        return parse_number(parser);
    
    print_error(parser, "unexpected character '%c'", parser->last_c);
    return NULL;
}

json_entry* json_parse_file(const char *filename)
{
    FILE *file = fopen(filename, "r");
    
    if (!file)
    {
        fprintf(stderr, PROGRAM_NAME": error: can't open '%s': %s\n", filename, strerror(errno));
        return NULL;
    }

    json_parser parser = {
        .file = file,
        .line = 1,
        .column = 0,
        .error = false
    };

    consume(&parser);
    skip_blank(&parser);

    json_entry *root = parse_entry(&parser);
    if (!parser.error && parser.last_c != EOF)
        print_error(&parser, "expected end of file, found '%c'", parser.last_c);

    fclose(file);

    if (parser.error)
    {
        json_free(root);
        return NULL;
    }

    return root;
}

// Access API

json_entry* json_object_get(const json_entry *object, const char *key)
{
    if (!object || object->type != JSON_OBJECT)
        return NULL;

    json_object *current = object->object;
    while (current)
    {
        if (!strcmp(current->key, key))
            return current->entry;
        current = current->next;
    }
    return NULL;
}

json_entry* json_array_get(const json_entry *array, size_t index)
{
    if (!array || array->type != JSON_ARRAY)
        return NULL;

    json_array *current = array->array;
    for (; index != 0 && current; --index)
        current = current->next;
    return current ? current->entry : NULL;
}

size_t json_array_count(const json_entry *array)
{
    if (!array || array->type != JSON_ARRAY)
        return 0;

    size_t count = 0;
    for (json_array *current = array->array; current; current = current->next)
        ++count;
    return count;
}


double json_as_number(const json_entry *entry)
{
    return entry->number;
}

bool json_try_as_number(const json_entry *entry, double *value)
{
    if (!entry || entry->type != JSON_NUMBER)
        return false;
    *value = json_as_number(entry);
    return true;
}


const char* json_as_string(const json_entry *entry)
{
    return entry->string;
}

bool json_try_as_string(const json_entry *entry, const char **string)
{
    if (!entry || entry->type != JSON_STRING)
        return false;
    *string = json_as_string(entry);
    return true;
}


bool json_as_bool(const json_entry *entry)
{
    return entry->boolean;
}

bool json_try_as_bool(const json_entry *entry, bool *value)
{
    if (!entry || entry->type != JSON_BOOL)
        return false;
    *value = json_as_bool(entry);
    return true;
}


json_type json_get_type(const json_entry *entry)
{
    return entry ? entry->type : JSON_NULL;
}

// Modification API

json_entry* json_new_entry()
{
    json_entry *entry = malloc(sizeof(json_entry));
    if (!entry)
        return NULL;
    *entry = (json_entry) {0};
    return entry;
}

bool json_object_set(json_entry *object, const char *key, json_entry *value)
{
    if (!object || object->type != JSON_OBJECT)
        return false;

    json_object *prev = NULL;
    json_object *current = object->object;
    while (current)
    {
        if (!strcmp(current->key, key))
        {
            json_free(current->entry);
            current->entry = value;
            return true;
        }
        prev = current;
        current = current->next;
    }

    char *key_copy = strdup(key);
    if (!key_copy)
        return false;

    json_object *new_object_entry = malloc(sizeof(json_object));
    if (!new_object_entry)
    {
        free(key_copy);
        return false;
    }
    new_object_entry->entry = value;
    new_object_entry->key = key_copy;
    new_object_entry->next = NULL;

    if (prev)
        prev->next = new_object_entry;
    else
        object->object = new_object_entry;
    return true;
}

json_entry* json_object_remove(json_entry *object, const char *key)
{
    if (!object || object->type != JSON_OBJECT)
        return NULL;

    json_object *prev = NULL;
    json_object *current = object->object;
    while (current)
    {
        if (!strcmp(current->key, key))
        {
            if (prev)
                prev->next = current->next;
            else
                object->object = current->next;
            
            json_entry *found_entry = current->entry;
            free(current->key);
            free(current);
            return found_entry;
        }
        prev = current;
        current = current->next;
    }
    return NULL;
}


bool json_array_append(json_entry *array, json_entry *value)
{
    if (!array || array->type != JSON_ARRAY)
        return false;

    json_array *last_array_entry = NULL;
    json_array *current = array->array;
    for (; current; current = current->next)
        last_array_entry = current;

    json_array *new_array_entry = malloc(sizeof(json_array));
    if (!new_array_entry)
        return false;
    new_array_entry->entry = value;
    new_array_entry->next = NULL;

    if (last_array_entry)
        last_array_entry->next = new_array_entry;
    else
        array->array = new_array_entry;
    return true;
}

bool json_array_insert(json_entry *array, size_t index, json_entry *value)
{
    if (!array || array->type != JSON_ARRAY)
        return false;

    json_array *prev = NULL;
    json_array *current = array->array;
    for (; index != 0 && current; --index)
    {
        prev = current;
        current = current->next;
    }

    json_array *new_array_entry = malloc(sizeof(json_array));
    if (!new_array_entry)
        return false;
    new_array_entry->entry = value;
    new_array_entry->next = current;

    if (prev)
        prev->next = new_array_entry;
    else
        array->array = new_array_entry;
    return true;
}

json_entry* json_array_remove(json_entry *array, size_t index)
{
    if (!array || array->type != JSON_ARRAY)
        return false;

    json_array *prev = NULL;
    json_array *current = array->array;
    for (; index != 0 && current; --index)
    {
        prev = current;
        current = current->next;
    }

    if (!current)
        return NULL;

    json_entry *entry = current->entry;
    if (prev)
        prev->next = current->next;
    else
        array->array = current->next;
    
    free(current);

    return entry;
}


void json_set_null(json_entry *entry)
{
    switch (entry->type)
    {
    case JSON_STRING:
        free(entry->string);
        break;

    case JSON_ARRAY:
        free_array(entry->array);
        break;

    case JSON_OBJECT:
        free_object(entry->object);
        break;
    
    default:
        break;
    }
    *entry = (json_entry) {0};
}

void json_set_number(json_entry *entry, double value)
{
    json_set_null(entry);
    entry->type = JSON_NUMBER;
    entry->number = value;
}

bool json_set_string(json_entry *entry, const char *new_string)
{
    char *string_copy = strdup(new_string);
    if (!string_copy)
        return false;

    json_set_null(entry);
    entry->type = JSON_STRING;
    entry->string = string_copy;
    return true;
}

void json_set_bool(json_entry *entry, bool value)
{
    json_set_null(entry);
    entry->type = JSON_BOOL;
    entry->boolean = value;
}

void json_set_object(json_entry *entry)
{
    json_set_null(entry);
    entry->type = JSON_OBJECT;
}

void json_set_array(json_entry *entry)
{
    json_set_null(entry);
    entry->type = JSON_ARRAY;
}

// Printing functions

static void fprint_indent(FILE *file, size_t indent)
{
    for (; indent != 0; --indent)
        fputs("  ", file);
}

static void fprint_array(FILE *file, const json_array *array, size_t depth)
{
    fputs("[\n", file);

    const json_array *entry = array;
    while (entry)
    {
        fprint_indent(file, depth+1);

        fprint_entry(file, entry->entry, depth+1);

        entry = entry->next;
        if (entry)
            fputs(",\n", file);
    }

    fputc('\n', file);
    fprint_indent(file, depth);
    fputc(']', file);
}

static void fprint_object(FILE *file, const json_object *object, size_t depth)
{
    fputs("{\n", file);

    const json_object *entry = object;
    while (entry)
    {
        fprint_indent(file, depth+1);

        fprintf(file, "\"%s\": ", entry->key);
        fprint_entry(file, entry->entry, depth+1);

        entry = entry->next;
        if (entry)
            fputs(",\n", file);
    }

    fputc('\n', file);
    fprint_indent(file, depth);
    fputc('}', file);
}

static void fprint_escape_string(FILE *file, char *str)
{
    for (; *str != '\0'; ++str)
    {
        if ((unsigned char)*str < 0x20 || *str == 0x7F)
        {
            fprintf(file, "\\u00%02X", *str & 0xFF);
            continue;
        }

        switch (*str)
        {
        case '"':  fputs("\\\"", file); break;
        case '\\': fputs("\\\\", file); break;
        case '/':  fputs("\\/",  file); break;
        case '\b': fputs("\\b",  file); break;
        case '\f': fputs("\\f",  file); break;
        case '\n': fputs("\\n",  file); break;
        case '\r': fputs("\\r",  file); break;
        case '\t': fputs("\\t",  file); break;
        default:   fputc(*str,   file); break;
        }
    }
}

static void fprint_entry(FILE *file, const json_entry *entry, size_t depth)
{
    if (!entry)
        return;

    switch (entry->type)
    {
    case JSON_NULL:
        fputs("null", file);
        break;

    case JSON_NUMBER:
        fprintf(file, "%g", entry->number);
        break;

    case JSON_STRING:
        fputc('"', file);
        fprint_escape_string(file, entry->string);
        fputc('"', file);
        break;

    case JSON_BOOL:
        fputs(entry->boolean ? "true" : "false", file);
        break;

    case JSON_ARRAY:
        fprint_array(file, entry->array, depth);
        break;

    case JSON_OBJECT:
        fprint_object(file, entry->object, depth);
        break;
    }
}

void json_fprint(FILE *file, const json_entry *entry)
{
    fprint_entry(file, entry, 0);
    fputc('\n', file);
}
