//https://leetcode.com/problems/trapping-rain-water/
class Solution {
public:
    int trap(vector<int>& height) {
        int n=height.size();
        vector<int>pre(n),suf(n);
        int ma=0;
        for(int i=0;i<n;++i){
            ma=max(ma,height[i]);
            pre[i]=ma;
        }
        ma=0;
        for(int i=n-1;i>=0;--i){
            ma=max(ma,height[i]);
            suf[i]=ma;
        }
        int ans=0;
        for(int i=0;i<n;++i){
            ans+=max(0,min(pre[i],suf[i])-height[i]);
        }
        return ans;
    }
};
