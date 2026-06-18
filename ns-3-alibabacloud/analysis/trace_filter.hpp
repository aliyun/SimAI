#ifndef TRACE_FILTER_HPP
#define TRACE_FILTER_HPP

#include <vector>
#include <stdint.h>
#include <cctype>
#include <regex>
#include <sstream>
#include "trace-format.h"

class TraceFilter{
public:
	/**********************************
	 * classes for test a single field
	 **********************************/
	class Field{
	public:
		uint32_t offset; // data offset in TraceFormat
		uint8_t op;

		Field(uint32_t _offset, std::string &_op){
			offset = _offset;
			if (_op == "=")
				op = 0;
			else if (_op == ">")
				op = 1;
			else if (_op == ">=")
				op = 2;
			else if (_op == "<")
				op = 3;
			else if (_op == "<=")
				op = 4;
			else if (_op == "!=")
				op = 5;
			else 
				op = 255;
		}
		std::string op_str(){
			if (op == 0)
				return "=";
			if (op == 1)
				return ">";
			if (op == 2)
				return ">=";
			if (op == 3)
				return "<";
			if (op == 4)
				return "<=";
			if (op == 5)
				return "!=";
			return "[Unknown op]";
		}
		virtual bool test(ns3::TraceFormat &tr) = 0;
		virtual std::string str() = 0;
	};
	#define OP(type) \
		do {\
			switch (op){\
				case 0: return *(type*)(((uint8_t*)&tr) + offset) == value;\
				case 1: return *(type*)(((uint8_t*)&tr) + offset) > value;\
				case 2: return *(type*)(((uint8_t*)&tr) + offset) >= value;\
				case 3: return *(type*)(((uint8_t*)&tr) + offset) < value;\
				case 4: return *(type*)(((uint8_t*)&tr) + offset) <= value;\
				case 5: return *(type*)(((uint8_t*)&tr) + offset) != value;\
				default: return false;\
			}\
		} while(0)
	class ByteField : public Field{
	public:
		uint8_t value;
		ByteField(uint32_t _offset, std::string &op, uint8_t _value) : Field(_offset, op), value(_value) {}
		virtual bool test(ns3::TraceFormat &tr){
			OP(uint8_t);
		}
		std::string str(){
			std::stringstream s;
			s << '[' << offset << ']' << op_str() << (int32_t)value;
			return s.str();
		}
	};
	class WordField : public Field{
	public:
		uint16_t value;
		WordField(uint32_t _offset, std::string &op, uint16_t _value) : Field(_offset, op), value(_value) {}
		virtual bool test(ns3::TraceFormat &tr){
			OP(uint16_t);
		}
		std::string str(){
			std::stringstream s;
			s << '[' << offset << ']' << op_str() << value;
			return s.str();
		}
	};
	class DwordField : public Field{
	public:
		uint32_t value;
		DwordField(uint32_t _offset, std::string &op, uint32_t _value) : Field(_offset, op), value(_value) {}
		virtual bool test(ns3::TraceFormat &tr){
			OP(uint32_t);
		}
		std::string str(){
			std::stringstream s;
			s << '[' << offset << ']' << op_str() << value;
			return s.str();
		}
	};
	class QwordField : public Field{
	public:
		uint64_t value;
		QwordField(uint32_t _offset, std::string &op, uint64_t _value) : Field(_offset, op), value(_value) {}
		virtual bool test(ns3::TraceFormat &tr){
			OP(uint64_t);
		}
		std::string str(){
			std::stringstream s;
			s << '[' << offset << ']' << op_str() << value;
			return s.str();
		}
	};

	class Node{
	public:
		uint32_t type; // node type: 0:expr, 1:&, 2:|
		Node* son[2];
		Field* f;

		Node(){
			son[0] = son[1] = 0;
			f = 0;
			type = 0;
		}
		void set_op(std::string op){
			if (op == "&")
				type = 1;
			else if (op == "|")
				type = 2;
		}
		bool test(ns3::TraceFormat &tr){
			if (type == 0){
				//printf("test %s\n", f->str().c_str());
				return f->test(tr);
			}
			if (type == 1)
				return son[0]->test(tr) && son[1]->test(tr);
			if (type == 2)
				return son[0]->test(tr) || son[1]->test(tr);
			return false;
		}
		void clear(){
			if (son[0]){
				son[0]->clear();
				delete son[0];
			}
			if (son[1]){
				son[1]->clear();
				delete son[1];
			}
			if (f)
				delete f;
		}
	};

	/***************
	 * members
	 ***************/
	Node* root;

	/******************
	 * methods
	 *****************/
	TraceFilter() : root(NULL){}
	// test a trace if it passes the filter
	bool test(ns3::TraceFormat &tr){
		if (root)
			return root->test(tr);
		return true;
	}

	// parse an filter expression
	void parse(std::string expr){
		root = _parse(expr);
	}
	// helper function: skip the spaces from idx i of str
	static void skip_space(uint32_t &i, const std::string &str){
		while (i < str.size() && isspace(str[i])) 
			i++;
	}
	// helper function: get a Field object from `field` `op` `value`
	Field* GetField(std::string field, std::string op, std::string value){
		Field *f = NULL;
		uint64_t v;
		sscanf(value.c_str(), "%li", &v);
		if (field == "time"){
			f = new QwordField(offsetof(ns3::TraceFormat, time), op, v);
		}else if (field == "node"){
			f = new WordField(offsetof(ns3::TraceFormat, node), op, v);
		}else if (field == "nodeType"){
			f = new ByteField(offsetof(ns3::TraceFormat, nodeType), op, v);
		}else if (field == "intf"){
			f = new ByteField(offsetof(ns3::TraceFormat, intf), op, v);
		}else if (field == "qidx"){
			f = new ByteField(offsetof(ns3::TraceFormat, qidx), op, v);
		}else if (field == "qlen"){
			f = new DwordField(offsetof(ns3::TraceFormat, qlen), op, v);
		}else if (field == "sip"){
			f = new DwordField(offsetof(ns3::TraceFormat, sip), op, v);
		}else if (field == "dip"){
			f = new DwordField(offsetof(ns3::TraceFormat, dip), op, v);
		}else if (field == "size"){
			f = new WordField(offsetof(ns3::TraceFormat, size), op, v);
		}else if (field == "l3Prot"){
			f = new ByteField(offsetof(ns3::TraceFormat, l3Prot), op, v);
		}else if (field == "event"){
			f = new ByteField(offsetof(ns3::TraceFormat, event), op, v);
		}else if (field == "ecn"){
			f = new ByteField(offsetof(ns3::TraceFormat, ecn), op, v);
		}else if (field == "data.sport"){
			f = new WordField(offsetof(ns3::TraceFormat, data.sport), op, v);
		}else if (field == "data.dport"){
			f = new WordField(offsetof(ns3::TraceFormat, data.dport), op, v);
		}else if (field == "data.seq"){
			f = new DwordField(offsetof(ns3::TraceFormat, data.seq), op, v);
		}else if (field == "ack.sport"){
			f = new WordField(offsetof(ns3::TraceFormat, ack.sport), op, v);
		}else if (field == "ack.dport"){
			f = new WordField(offsetof(ns3::TraceFormat, ack.dport), op, v);
		}else if (field == "ack.flags"){
			f = new ByteField(offsetof(ns3::TraceFormat, ack.flags), op, v);
		}else if (field == "qp.sport"){
			f = new WordField(offsetof(ns3::TraceFormat, qp.sport), op, v);
		}else if (field == "qp.dport"){
			f = new WordField(offsetof(ns3::TraceFormat, qp.dport), op, v);
		}
		return f;
	}
	
	#define OP_COMPARE "=|>|>=|<|<=|!="
	#define BASE_EXPR_REGEX "\\s*([a-zA-Z0-9\\.]+)\\s*(" OP_COMPARE")\\s*([x,[:xdigit:]]+)\\s*"
	// the real implementation of parse, allows recursion
	Node* _parse(std::string expr){
		expr = strip_outer_bracket(expr);
		Node* res = NULL;
		std::smatch m;
		if (std::regex_match(expr, m, std::regex(BASE_EXPR_REGEX))){ // a base expression
			std::string field = m[1].str();
			std::string op = m[2].str();
			std::string value = m[3].str();
			Field *f = GetField(field, op, value);
			if (f){
				res = new Node();
				res->f = f;
				res->type = 0;
			}else{ // maybe this is a short hand
				res = parse_shorthand(field, op, value);
			}
		}else {
			Node *left = NULL;
			std::string op_str, right_str;
			if (std::regex_match(expr, m, std::regex(BASE_EXPR_REGEX"(&|\\|)(.*)"))){ // a base express &| other things
				// get left
				std::string field = m[1].str();
				std::string op = m[2].str();
				std::string value = m[3].str();
				Field *f = GetField(field, op, value);
				if (f){
					left = new Node();
					left->f = f;
				}else { // maybe this is a short hand
					left = parse_shorthand(field, op, value);
					if (left == NULL)
						return NULL;
				}
				// assign right str
				right_str = m[5].str();
				op_str = m[4].str();
			}else { // (base expression) &| other things
				uint32_t start, i = 0;
				// get left
				skip_space(i, expr);
				if (i >= expr.size())
					return NULL;
				start = i;
				if (expr[i] == '('){
					// find matching brackets
					uint32_t c = 1;
					for (i++; i < expr.size() && (expr[i] != ')' || c > 1); i++){
						if (expr[i] == '(')
							c++;
						else if (expr[i] == ')')
							c--;
					}
					if (i >= expr.size())
						return NULL;
					i++;
					left = _parse(expr.substr(start + 1, i - start - 2));
				}
				if (!left)
					return NULL;
				// assign right str
				std::string s = expr.substr(i, expr.size() - i);
				if (std::regex_match(s, m, std::regex("\\s*(&|\\|)(.*)\\s*"))){
					right_str = m[2].str();
					op_str = m[1].str();
				}
			}
			// get right
			Node *right = _parse(right_str);
			if (right){
				res = new Node;
				res->son[0] = left;
				res->son[1] = right;
				res->set_op(op_str);
			}else {
				left->clear();
				delete left;
			}
		}
		return res;
	}
	std::string strip_outer_bracket(std::string expr){
		uint32_t i = 0, start, end;
		skip_space(i, expr);
		// if begin with '('
		if (expr[i] == '('){
			start = i+1;
			uint32_t c = 1;
			for (i++; i < expr.size() && (expr[i] != ')' || c > 1); i++){
				if (expr[i] == '(')
					c++;
				else if (expr[i] == ')')
					c--;
			}
			// if cannot find matching ')', return original
			if (i >= expr.size())
				return expr;
			// matching ')' is at i
			end = i;
			// skip the spaces, see if we can reach the end of the string
			i++;
			skip_space(i, expr);
			// rest of the string are spaces, this is a pair of outer bracket
			if (i >= expr.size())
				return strip_outer_bracket(expr.substr(start, end-start)); // recursively strip brackets
		}
		return expr;
	}
	Node* parse_shorthand(std::string shorthand, std::string op, std::string value){
		if (shorthand == "flow" || shorthand == "biflow" || shorthand == "rflow"){ // forward flow, bi-directional flow, reverse flow
			// parse 4-tuples here
			uint32_t sip, dip;
			uint16_t sport, dport;
			if (op != "=") // using flow shorthand, must use '='
				return NULL;
			if (sscanf(value.c_str(), "%i,%i,%hu,%hu", &sip, &dip, &sport, &dport) == 4){
				char buf[512];
				if (shorthand == "flow"){
					sprintf(buf, "sip=%u&dip=%u&((l3Prot=17&data.sport=%hu&data.dport=%hu)|((l3Prot=0xFC|l3Prot=0xFD)&ack.sport=%hu&ack.dport=%hu)|(l3Prot=0x0&qp.sport=%hu&qp.dport=%hu))", sip, dip, sport, dport, sport, dport, sport, dport);
					return _parse(buf);
				}else if (shorthand == "biflow"){
					sprintf(buf, "(sip=%u&dip=%u&((l3Prot=17&data.sport=%hu&data.dport=%hu)|((l3Prot=0xFC|l3Prot=0xFD)&ack.sport=%hu&ack.dport=%hu)|(l3Prot=0x0&qp.sport=%hu&qp.dport=%hu)))|(sip=%u&dip=%u&((l3Prot=17&data.sport=%hu&data.dport=%hu)|((l3Prot=0xFC|l3Prot=0xFD)&ack.sport=%hu&ack.dport=%hu)|(l3Prot=0x0&qp.sport=%hu&qp.dport=%hu)))", sip, dip, sport, dport, sport, dport, sport, dport, dip, sip, dport, sport, dport, sport, dport, sport);
					return _parse(buf);
				}else if (shorthand == "rflow"){
					sprintf(buf, "sip=%u&dip=%u&((l3Prot=17&data.sport=%hu&data.dport=%hu)|((l3Prot=0xFC|l3Prot=0xFD)&ack.sport=%hu&ack.dport=%hu)|(l3Prot=0x0&qp.sport=%hu&qp.dport=%hu))", dip, sip, dport, sport, dport, sport, dport, sport);
					return _parse(buf);
				}
			}
		}else if (shorthand == "queue"){
			// parse "node,intf,qidx"
			uint16_t node;
			uint8_t intf, qidx;
			if (op != "=") // using queue shorthand, must use '='
				return NULL;
			if (sscanf(value.c_str(), "%hu,%hhu,%hhu", &node, &intf,&qidx) == 3){
				char buf[512];
				sprintf(buf, "node=%u&intf=%u&qidx=%u", node, intf, qidx);
				return _parse(buf);
			}
		}
		return NULL;
	}
	// print the expression
	std::string str(){
		return str(root);
	}
	std::string str(Node* n){
		if (n == NULL)
			return "";
		if (n->type == 0)
			return n->f->str();
		return '(' + str(n->son[0]) + ')' + (n->type == 1? '&' : '|') + '(' + str(n->son[1]) + ')';
	}
	#undef OP_COMPARE
	#undef BASE_EXPR_REGEX
};
#endif /* TRACE_FILTER_HPP */
