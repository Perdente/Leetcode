class Solution {
public:
    bool checkValidString(string s) {
        int cnt;
        bool flag=true;
        for(int repeat=0;repeat<2;++repeat){
            cnt=0;
            for(auto i:s){
                if(i=='(' or i=='*')cnt++;
                else cnt--;
                if(cnt==-1){
                    flag=false;
                }
            }
            reverse(s.begin(),s.end());
            for(auto &i:s){
                if( i=='(' )i=')';
                else if( i==')' )i='(';
            }
        }
        return flag;
    }
};
