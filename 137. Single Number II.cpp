class Solution {
public:
    int singleNumber(vector<int>& nums) {
        unordered_map<int,int>mp;
        for(auto i:nums)mp[i]++;
        int ans=-1;
        for(auto i:mp){
          if(i.second==1){
            ans=i.first;break;
          } 
        }
        return ans;
    }
};
