class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        const int inf=1e9;
        int H=grid.size();
        int W=grid[0].size();
        vector<vector<int>>dp(H,vector<int>(W));
        for(int i=0;i<H;++i){
            for(int j=0;j<W;++j){
                if(i==0 and j==0){
                    dp[i][j]=grid[i][j];
                    continue;
                }
                dp[i][j]=min((i==0?inf:dp[i-1][j]),(j==0?inf:dp[i][j-1]))+grid[i][j];
            }
        }
        return dp[H-1][W-1];
    }
};
