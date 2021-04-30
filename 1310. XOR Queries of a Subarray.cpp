class Solution {
public:
    vector<int> xorQueries(vector<int>& arr, vector<vector<int>>& queries) {
       
        int temp=0,n=arr.size(),m=queries.size();
        vector<int>prefix(n);
        for(int i=0;i<n;++i){
          temp^=arr[i];
          prefix[i]=temp;
        }
        vector<int>ans;
        for(int i=0;i<m;++i){
           int l=queries[i][0],r=queries[i][1];
           if(l)ans.push_back(prefix[r]^prefix[l-1]);
           else ans.push_back(prefix[r]);
        } 
        return ans;
    }
};
