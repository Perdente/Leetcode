class Solution {
public:
    vector<vector<int>> matrixBlockSum(vector<vector<int>>& mat, int k) {
        int n=mat.size();
        int m=mat[0].size();
        vector<vector<int>> sum(n,vector<int>(m,0));
        for(int i=0;i<n;++i){
            for(int j=0;j<m;++j){
                for(int x=-k;x<=k;++x){
                    if(0<=i+x and i+x<n){
                        int l=max(0,j-k);
                        int r=j+k+1;
                        sum[i+x][l]+=mat[i][j];
                        if(r<m)sum[i+x][r]-=mat[i][j];
                    }
                }
            }
        }
        int total=0;
        for(int i=0;i<n;++i){
            total=0;
            for(int j=0;j<m;++j){
                total+=sum[i][j];
                sum[i][j]=total;
            }
        }
        return sum;
    }
};
