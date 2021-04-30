class Solution {
public:
    vector<string> findRepeatedDnaSequences(string s) {
        int n=s.size();
        unordered_map<string,int>mp;
        for(int i=0;i+9<n;++i){
            string str=s.substr(i,10);
            mp[str]++;
        }
        vector<string>ans;
        for(auto i:mp){
            if(i.second>1)ans.push_back(i.first);
        }
        return ans;
    }
   
};
