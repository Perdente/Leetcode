class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        int H=matrix.size();
        int W=matrix[0].size();
        if(H==0 or W==0)
            return 0;
        vector<vector<int>>dp(H,vector<int>(W));
        int ans=0;
        for(int row=0;row<H;++row){
            for(int col=0;col<W;++col){
                if(matrix[row][col]=='1'){
                    dp[row][col]=1;
                    if(row>0 and col>0)dp[row][col]+=min({dp[row-1][col],dp[row][col-1],dp[row-1][col-1]});
                }
                ans=max(ans,dp[row][col]);
            }
            
        }
        return ans*ans;
    }
};
