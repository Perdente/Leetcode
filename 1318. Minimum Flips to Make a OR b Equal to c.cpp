class Solution {
public:
    int minFlips(int a, int b, int c) {
        int cnt=0;
        for(int i=0;i<32;++i){
            bool flag=false,flag1=false,flag2=false;
            if(a&(1<<i))flag=true;
            if(b&(1<<i))flag1=true;
            if(c&(1<<i))flag2=true;
            if(flag2){
                if(!flag and !flag1)cnt++;
            }else{
                if(flag)cnt++;
                if(flag1)cnt++;
            }
        }
        return cnt;
    }
};
