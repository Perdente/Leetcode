class Solution {
public:
    string longestWord(vector<string>& words) {
        
        set<string>dictionary(words.begin(),words.end());
        set<string>good;
        
        sort(words.begin(),words.end(),[&](const string& a,const string& b){
            return a.length()<b.length();
        });
        
        string str="";
        auto check=[&](const string& s){
            if(s.length() > str.length()){
                str=s;
            }
            else if(s<str){
                str=s;
            }
        };
        
        
        for(string s:words){
            if((int)s.length()==1){good.insert(s);check(s);continue;}
            string shorten=s;
            shorten.pop_back();
            if(dictionary.count(shorten) and good.count(shorten)){
                check(s);
                good.insert(s);
            }
        }
        return str;
    }
    
};
