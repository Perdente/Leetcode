class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>>ans;
        int n=nums.size();
        ans.resize(1<<n);
        for(int mask=0;mask<(1<<n);++mask){
            for(int i=0;i<n;++i){
                if(mask&(1<<i))ans[mask].push_back(nums[i]);
            }
        }
        return ans;
        
    }
};
