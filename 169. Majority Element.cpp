class Solution {
public:
    int majorityElement(vector<int>& nums) {
        unordered_map<int,int>mp;
        for(auto i:nums){
            mp[i]++;
        }
        int ma=0,ans=-1;
        for(auto i:mp){
            if(i.second>ma){
                ma=i.second;
                ans=i.first;
            }
        }
        return ans;
    }
};
