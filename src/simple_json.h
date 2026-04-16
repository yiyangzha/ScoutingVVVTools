#ifndef SIMPLE_JSON_H
#define SIMPLE_JSON_H

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simple_json {

class Value {
public:
    enum class Type {
        Null,
        Bool,
        Number,
        String,
        Array,
        Object,
    };

    using Array = std::vector<Value>;
    using Object = std::map<std::string, Value>;

    Value() = default;

    static Value makeBool(bool value) {
        Value out;
        out.type_ = Type::Bool;
        out.boolValue_ = value;
        return out;
    }

    static Value makeNumber(long double value) {
        Value out;
        out.type_ = Type::Number;
        out.numberValue_ = value;
        return out;
    }

    static Value makeString(std::string value) {
        Value out;
        out.type_ = Type::String;
        out.stringValue_ = std::move(value);
        return out;
    }

    static Value makeArray(Array value) {
        Value out;
        out.type_ = Type::Array;
        out.arrayValue_ = std::move(value);
        return out;
    }

    static Value makeObject(Object value) {
        Value out;
        out.type_ = Type::Object;
        out.objectValue_ = std::move(value);
        return out;
    }

    Type type() const {
        return type_;
    }

    bool isNull() const {
        return type_ == Type::Null;
    }

    bool isBool() const {
        return type_ == Type::Bool;
    }

    bool isNumber() const {
        return type_ == Type::Number;
    }

    bool isString() const {
        return type_ == Type::String;
    }

    bool isArray() const {
        return type_ == Type::Array;
    }

    bool isObject() const {
        return type_ == Type::Object;
    }

    bool asBool() const {
        requireType(Type::Bool, "bool");
        return boolValue_;
    }

    long double asNumber() const {
        requireType(Type::Number, "number");
        return numberValue_;
    }

    int asInt() const {
        return static_cast<int>(asNumber());
    }

    std::string asString() const {
        requireType(Type::String, "string");
        return stringValue_;
    }

    const Array& asArray() const {
        requireType(Type::Array, "array");
        return arrayValue_;
    }

    const Object& asObject() const {
        requireType(Type::Object, "object");
        return objectValue_;
    }

    bool contains(const std::string& key) const {
        if (!isObject()) {
            return false;
        }
        return objectValue_.find(key) != objectValue_.end();
    }

    const Value* find(const std::string& key) const {
        if (!isObject()) {
            return nullptr;
        }
        const auto it = objectValue_.find(key);
        if (it == objectValue_.end()) {
            return nullptr;
        }
        return &it->second;
    }

    const Value& at(const std::string& key) const {
        const Value* child = find(key);
        if (child == nullptr) {
            throw std::runtime_error("Missing JSON key: " + key);
        }
        return *child;
    }

    std::string getStringOr(const std::string& key, const std::string& fallback) const {
        const Value* child = find(key);
        return child == nullptr ? fallback : child->asString();
    }

    int getIntOr(const std::string& key, int fallback) const {
        const Value* child = find(key);
        return child == nullptr ? fallback : child->asInt();
    }

    bool getBoolOr(const std::string& key, bool fallback) const {
        const Value* child = find(key);
        return child == nullptr ? fallback : child->asBool();
    }

    long double getNumberOr(const std::string& key, long double fallback) const {
        const Value* child = find(key);
        return child == nullptr ? fallback : child->asNumber();
    }

    std::vector<std::string> toStringArray() const {
        std::vector<std::string> out;
        for (const auto& item : asArray()) {
            out.push_back(item.asString());
        }
        return out;
    }

private:
    void requireType(Type expected, const char* name) const {
        if (type_ != expected) {
            throw std::runtime_error(std::string("JSON value is not a ") + name);
        }
    }

    Type type_ = Type::Null;
    bool boolValue_ = false;
    long double numberValue_ = 0.;
    std::string stringValue_;
    Array arrayValue_;
    Object objectValue_;
};

namespace detail {

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text) {}

    Value parse() {
        skipWhitespace();
        Value value = parseValue();
        skipWhitespace();
        if (!eof()) {
            fail("Unexpected trailing content");
        }
        return value;
    }

private:
    Value parseValue() {
        if (eof()) {
            fail("Unexpected end of JSON");
        }

        const char ch = peek();
        if (ch == '{') {
            return parseObject();
        }
        if (ch == '[') {
            return parseArray();
        }
        if (ch == '"') {
            return Value::makeString(parseString());
        }
        if (ch == 't') {
            expectLiteral("true");
            return Value::makeBool(true);
        }
        if (ch == 'f') {
            expectLiteral("false");
            return Value::makeBool(false);
        }
        if (ch == 'n') {
            expectLiteral("null");
            return Value();
        }
        if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch)) != 0) {
            return Value::makeNumber(parseNumber());
        }

        fail("Unexpected character while parsing JSON value");
        return Value();
    }

    Value parseObject() {
        consume('{');
        skipWhitespace();

        Value::Object object;
        if (peekIf('}')) {
            consume('}');
            return Value::makeObject(std::move(object));
        }

        while (true) {
            skipWhitespace();
            if (peek() != '"') {
                fail("Expected string key in JSON object");
            }
            const std::string key = parseString();

            skipWhitespace();
            consume(':');
            skipWhitespace();
            object.emplace(key, parseValue());
            skipWhitespace();

            if (peekIf('}')) {
                consume('}');
                break;
            }
            consume(',');
            skipWhitespace();
        }

        return Value::makeObject(std::move(object));
    }

    Value parseArray() {
        consume('[');
        skipWhitespace();

        Value::Array array;
        if (peekIf(']')) {
            consume(']');
            return Value::makeArray(std::move(array));
        }

        while (true) {
            skipWhitespace();
            array.push_back(parseValue());
            skipWhitespace();

            if (peekIf(']')) {
                consume(']');
                break;
            }
            consume(',');
            skipWhitespace();
        }

        return Value::makeArray(std::move(array));
    }

    std::string parseString() {
        consume('"');
        std::string out;

        while (!eof()) {
            const char ch = get();
            if (ch == '"') {
                return out;
            }
            if (ch == '\\') {
                if (eof()) {
                    fail("Incomplete JSON escape sequence");
                }
                const char escaped = get();
                switch (escaped) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': appendCodePoint(out, parseHexCodePoint()); break;
                    default: fail("Unsupported JSON escape sequence");
                }
                continue;
            }
            if (static_cast<unsigned char>(ch) < 0x20) {
                fail("Control character in JSON string");
            }
            out.push_back(ch);
        }

        fail("Unterminated JSON string");
        return std::string();
    }

    long double parseNumber() {
        const size_t begin = pos_;

        if (peekIf('-')) {
            ++pos_;
        }

        if (peekIf('0')) {
            ++pos_;
        } else {
            requireDigits();
        }

        if (peekIf('.')) {
            ++pos_;
            requireDigits();
        }

        if (peekIf('e') || peekIf('E')) {
            ++pos_;
            if (peekIf('+') || peekIf('-')) {
                ++pos_;
            }
            requireDigits();
        }

        const std::string token = text_.substr(begin, pos_ - begin);
        char* end = nullptr;
        const long double value = std::strtold(token.c_str(), &end);
        if (end == nullptr || *end != '\0') {
            fail("Invalid JSON number");
        }
        return value;
    }

    void requireDigits() {
        if (eof() || std::isdigit(static_cast<unsigned char>(peek())) == 0) {
            fail("Expected digits in JSON number");
        }
        while (!eof() && std::isdigit(static_cast<unsigned char>(peek())) != 0) {
            ++pos_;
        }
    }

    unsigned parseHexCodePoint() {
        unsigned value = 0;
        for (int i = 0; i < 4; ++i) {
            if (eof()) {
                fail("Incomplete unicode escape");
            }
            const char ch = get();
            value <<= 4;
            if (ch >= '0' && ch <= '9') {
                value |= static_cast<unsigned>(ch - '0');
            } else if (ch >= 'a' && ch <= 'f') {
                value |= static_cast<unsigned>(10 + ch - 'a');
            } else if (ch >= 'A' && ch <= 'F') {
                value |= static_cast<unsigned>(10 + ch - 'A');
            } else {
                fail("Invalid unicode escape");
            }
        }
        return value;
    }

    static void appendCodePoint(std::string& out, unsigned codePoint) {
        if (codePoint <= 0x7F) {
            out.push_back(static_cast<char>(codePoint));
            return;
        }
        if (codePoint <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | ((codePoint >> 6) & 0x1F)));
            out.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
            return;
        }
        out.push_back(static_cast<char>(0xE0 | ((codePoint >> 12) & 0x0F)));
        out.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    }

    void expectLiteral(const char* literal) {
        while (*literal != '\0') {
            if (eof() || get() != *literal) {
                fail("Invalid JSON literal");
            }
            ++literal;
        }
    }

    void skipWhitespace() {
        while (!eof() && std::isspace(static_cast<unsigned char>(peek())) != 0) {
            ++pos_;
        }
    }

    bool eof() const {
        return pos_ >= text_.size();
    }

    char peek() const {
        if (eof()) {
            return '\0';
        }
        return text_[pos_];
    }

    bool peekIf(char expected) const {
        return !eof() && text_[pos_] == expected;
    }

    char get() {
        if (eof()) {
            fail("Unexpected end of JSON");
        }
        return text_[pos_++];
    }

    void consume(char expected) {
        if (get() != expected) {
            fail(std::string("Expected '") + expected + "' in JSON");
        }
    }

    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error(message + " at JSON offset " + std::to_string(pos_));
    }

    const std::string& text_;
    size_t pos_ = 0;
};

}  // namespace detail

inline Value parse(const std::string& text) {
    return detail::Parser(text).parse();
}

inline Value parseFile(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("Failed to open JSON file: " + path);
    }

    std::string text((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    return parse(text);
}

}  // namespace simple_json

#endif
