class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int n=nums.size();
        int ans=0;
        int pref=0;
        unordered_map<int,int>count;
        count[pref]++;
        for(int r=0;r<n;++r){
            //k=pref[r]-pref[l-1]
            //pref[l-1]=pref[r]-k
            pref+=nums[r];
            int need=pref-k;
            ans+=count[need];
            count[pref]++;
        }
        return ans;
    }
};
