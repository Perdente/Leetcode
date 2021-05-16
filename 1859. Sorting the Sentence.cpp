//https://leetcode.com/contest/biweekly-contest-52/problems/sorting-the-sentence/
class Solution {
public:
    string sortSentence(string s) {
        map<int,string>mp;
        int n=s.size();
        for(int i=0;i<n;++i){
            if(isdigit(s[i])){

                int j=i,cnt=0;
                while(j>0 and s[j-1]!=' ')j--,cnt++;
                string str=s.substr(j,cnt);
                int digit=s[i]-'0';
                mp[digit]=str;
            }
        }
        string ans="";
        for(auto i:mp){
            ans+=i.second;
            ans+=(char)' ';
        }
        n=ans.size();
        ans=ans.substr(0,n-1);
        return ans;
    }
};
