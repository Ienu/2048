using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Collections;
using System.Threading;

namespace _2048
{
    public partial class Form1 : Form
    {
        public class Board
        {
            private int[,] data = new int[4, 4];

            public void GetData(int[,] board)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        board[i, j] = data[i, j];
                    }
                }
            }
            public void SetData(int[,] board)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        data[i, j] = board[i, j];
                    }
                }
            }
            public bool IsEqual(Board board)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        if (board.data[i, j] != data[i, j])
                        {
                            return false;
                        }
                    }
                }
                return true;
            }
            public void Record(Board bd)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        bd.data[i, j] = data[i, j];
                    }
                }
            }
            public void Restore(Board bd)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        data[i, j] = bd.data[i, j];
                    }
                }
            }
            public void Left()
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        if (data[i, j] == 0)
                        {
                            for (int k = j + 1; k < 4; k++)
                            {
                                if (data[i, k] != 0)
                                {
                                    data[i, j] = data[i, k];
                                    data[i, k] = 0;
                                    break;
                                }
                            }
                        }
                    }
                    for (int j = 0; j < 3; j++)
                    {
                        if (data[i, j] == data[i, j + 1] && data[i, j] != 0)
                        {
                            data[i, j] *= 2;
                            for (int k = j + 1; k < 3; k++)
                            {
                                data[i, k] = data[i, k + 1];
                            }
                            data[i, 3] = 0;
                        }
                    }
                }
            }
            public void Right()
            {
                for (int i = 3; i >= 0; i--)
                {
                    for (int j = 3; j >= 1; j--)
                    {
                        if (data[i, j] == 0)
                        {
                            for (int k = j - 1; k >= 0; k--)
                            {
                                if (data[i, k] != 0)
                                {
                                    data[i, j] = data[i, k];
                                    data[i, k] = 0;
                                    break;
                                }
                            }
                        }
                    }
                    for (int j = 3; j >= 1; j--)
                    {
                        if (data[i, j] == data[i, j - 1] && data[i, j] != 0)
                        {
                            data[i, j] *= 2;
                            for (int k = j - 1; k >= 1; k--)
                            {
                                data[i, k] = data[i, k - 1];
                            }
                            data[i, 0] = 0;
                        }
                    }
                }
            }
            public void Up()
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        if (data[j, i] == 0)
                        {
                            for (int k = j + 1; k < 4; k++)
                            {
                                if (data[k, i] != 0)
                                {
                                    data[j, i] = data[k, i];
                                    data[k, i] = 0;
                                    break;
                                }
                            }
                        }
                    }
                    for (int j = 0; j < 3; j++)
                    {
                        if (data[j, i] == data[j + 1, i] && data[j, i] != 0)
                        {
                            data[j, i] *= 2;
                            for (int k = j + 1; k < 3; k++)
                            {
                                data[k, i] = data[k + 1, i];
                            }
                            data[3, i] = 0;
                        }
                    }
                }
            }
            public void Down()
            {
                for (int i = 3; i >= 0; i--)
                {
                    for (int j = 3; j >= 1; j--)
                    {
                        if (data[j, i] == 0)
                        {
                            for (int k = j - 1; k >= 0; k--)
                            {
                                if (data[k, i] != 0)
                                {
                                    data[j, i] = data[k, i];
                                    data[k, i] = 0;
                                    break;
                                }
                            }
                        }
                    }
                    for (int j = 3; j >= 1; j--)
                    {
                        if (data[j, i] == data[j - 1, i] && data[j, i] != 0)
                        {
                            data[j, i] *= 2;
                            for (int k = j - 1; k >= 1; k--)
                            {
                                data[k, i] = data[k - 1, i];
                            }
                            data[0, i] = 0;
                        }
                    }
                }
            }
            public void GetBlank(List<Point> lp)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        if (data[i, j] == 0)
                        {
                            lp.Add(new Point(j, i));
                        }
                    }
                }
            }
        };
        //private static Board fbd = new Board();
        private static int[,] board = new int[4, 4];
        #region defination of color
        private Color BackgroundColor = Color.FromArgb(205, 193, 180);
        private Color Color2 = Color.FromArgb(238, 228, 218);
        private Color Color4 = Color.FromArgb(237, 224, 200);
        private Color Color8 = Color.FromArgb(242, 177, 121);
        private Color Color16 = Color.FromArgb(245, 149, 99);
        private Color Color32 = Color.FromArgb(246, 124, 95);
        private Color Color64 = Color.FromArgb(246, 98, 61);
        private Color Color128 = Color.FromArgb(237, 207, 114);
        private Color Color256 = Color.FromArgb(236, 203, 97);
        private Color Color512 = Color.FromArgb(237, 200, 80);
        private Color Color1024 = Color.FromArgb(237, 197, 63);
        private Color Color2048 = Color.FromArgb(237, 194, 46);
        #endregion
        #region defination of size
        private Point Org = new Point(64, 237);
        private int height = 105;
        private int width = 105;
        private int nheight = 121;
        private int nwidth = 121;
        #endregion

        private static bool b_auto = false;
        private static bool b_paint = true;
        private static Form1 fm;
        private Thread th = new Thread(new ThreadStart(ThreadProc));

        public static void ThreadProc()
        {
            while (true)
            {
                if (b_auto)
                {
                    KeyEventArgs kea = new KeyEventArgs(Keys.Right);
                    /*fm.Form1_KeyUp(fm, kea);
                    Thread.Sleep(200);*/

                    if (board[0, 0] != 0)
                    {
                        kea = new KeyEventArgs(Keys.Up);
                        fm.Form1_KeyUp(fm, kea);
                        Thread.Sleep(200);
                    }
                    for (int i = 0; i < 10; i++)
                    {
                        kea = new KeyEventArgs(Keys.Left);
                        fm.Form1_KeyUp(fm, kea);
                        Thread.Sleep(20);
                        kea = new KeyEventArgs(Keys.Up);
                        fm.Form1_KeyUp(fm, kea);
                        Thread.Sleep(200);
                    }
                }
            }
        }

        private Point RndPoint()
        {
            Point p = new Point(-1, -1);
            Random rd = new Random();
            List<Point> lp = new List<Point>(16);
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    if (board[i, j] == 0)
                    {
                        lp.Add(new Point(j, i));
                    }
                }
            }
            if (lp.Count > 0)
            {
                int index = rd.Next(0, lp.Count);
                return lp[index];
            }
            return p;
        }
        private int RndValue()
        {
            Random rd = new Random();
            int p = rd.Next(0, 9);
            if (p == 0)
            {
                return 4;
            }
            return 2;
        }
        private void PaintRefresh()
        {
            Graphics g = this.CreateGraphics();
            Pen myPen = new Pen(Color.Red, 4);
            SolidBrush mybrush = new SolidBrush(Color.Blue);
            Font drawFont = new Font("Vrinda", 30, FontStyle.Bold);

            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    int posX = Org.X + j * nwidth;
                    int posY = Org.Y + i * nheight;

                    if (board[i, j] != 0)
                    {
                        Color tColor = new Color();
                        Color txtColor = Color.FromArgb(249, 246, 242);
                        int fontsize = 30;
                        switch (board[i, j])
                        {
                            case 2:
                                tColor = Color2;
                                txtColor = Color.FromArgb(119, 110, 101);
                                fontsize = 50;
                                break;
                            case 4:
                                tColor = Color4;
                                txtColor = Color.FromArgb(119, 110, 101);
                                fontsize = 50;
                                break;
                            case 8:
                                tColor = Color8;
                                fontsize = 50;
                                break;
                            case 16:
                                tColor = Color16;
                                fontsize = 50;
                                break;
                            case 32:
                                tColor = Color32;
                                fontsize = 50;
                                break;
                            case 64:
                                tColor = Color64;
                                fontsize = 50;
                                break;
                            case 128:
                                tColor = Color128;
                                fontsize = 40;
                                break;
                            case 256:
                                tColor = Color256;
                                fontsize = 40;
                                break;
                            case 512:
                                tColor = Color512;
                                fontsize = 40;
                                break;
                            case 1024:
                                tColor = Color1024;
                                fontsize = 30;
                                break;
                            case 2048:
                                tColor = Color2048;
                                fontsize = 30;
                                break;
                            default:
                                tColor = Color.Black;
                                fontsize = 20;
                                break;
                        }
                        drawFont = new Font("Vrinda", fontsize, FontStyle.Bold);

                        Rectangle rect = new Rectangle(posX, posY, width, height);
                        g.FillRectangle(new SolidBrush(tColor), rect);

                        string dstring = board[i, j].ToString();
                        StringFormat sf = new StringFormat();
                        sf.Alignment = StringAlignment.Center;
                        sf.LineAlignment = StringAlignment.Center;
                        g.DrawString(dstring, drawFont, new SolidBrush(txtColor), rect, sf);
                    }
                    else
                    {
                        g.FillRectangle(new SolidBrush(BackgroundColor),
                            new Rectangle(posX, posY, width, height));
                    }
                }
            }
            myPen.Dispose();
            mybrush.Dispose();
            g.Dispose();
        }
        public Form1()
        {
            InitializeComponent();
        }
        private void Form1_Paint(object sender, PaintEventArgs e)
        {
            PaintRefresh();
        }
        private void Form1_KeyUp(object sender, KeyEventArgs e)
        {
            bool combine = false;
            switch (e.KeyCode)
            {
                #region defination of direction key
                case Keys.Up:
                    for (int i = 0; i < 4; i++)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            if (board[j, i] == 0)
                            {
                                for (int k = j + 1; k < 4; k++)
                                {
                                    if (board[k, i] != 0)
                                    {
                                        board[j, i] = board[k, i];
                                        board[k, i] = 0;
                                        combine = true;
                                        break;
                                    }
                                }
                            }
                        }
                        for (int j = 0; j < 3; j++)
                        {
                            if (board[j, i] == board[j + 1, i] && board[j, i] != 0)
                            {
                                board[j, i] *= 2;
                                combine = true;
                                for (int k = j + 1; k < 3; k++)
                                {
                                    board[k, i] = board[k + 1, i];
                                }
                                board[3, i] = 0;
                            }
                        }
                    }
                    break;
                case Keys.Down:
                    for (int i = 3; i >= 0; i--)
                    {
                        for (int j = 3; j >= 1; j--)
                        {
                            if (board[j, i] == 0)
                            {
                                for (int k = j - 1; k >= 0; k--)
                                {
                                    if (board[k, i] != 0)
                                    {
                                        board[j, i] = board[k, i];
                                        board[k, i] = 0;
                                        combine = true;
                                        break;
                                    }
                                }
                            }
                        }
                        for (int j = 3; j >= 1; j--)
                        {
                            if (board[j, i] == board[j - 1, i] && board[j, i] != 0)
                            {
                                board[j, i] *= 2;
                                combine = true;
                                for (int k = j - 1; k >= 1; k--)
                                {
                                    board[k, i] = board[k - 1, i];
                                }
                                board[0, i] = 0;
                            }
                        }
                    }
                    break;
                case Keys.Left:
                    for (int i = 0; i < 4; i++)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            if (board[i, j] == 0)
                            {
                                for (int k = j + 1; k < 4; k++)
                                {
                                    if (board[i, k] != 0)
                                    {
                                        board[i, j] = board[i, k];
                                        board[i, k] = 0;
                                        combine = true;
                                        break;
                                    }
                                }
                            }
                        }
                        for (int j = 0; j < 3; j++)
                        {
                            if (board[i, j] == board[i, j + 1] && board[i, j] != 0)
                            {
                                board[i, j] *= 2;
                                combine = true;
                                for (int k = j + 1; k < 3; k++)
                                {
                                    board[i, k] = board[i, k + 1];
                                }
                                board[i, 3] = 0;
                            }
                        }
                    }
                    break;
                case Keys.Right:
                    for (int i = 3; i >= 0; i--)
                    {
                        for (int j = 3; j >= 1; j--)
                        {
                            if (board[i, j] == 0)
                            {
                                for (int k = j - 1; k >= 0; k--)
                                {
                                    if (board[i, k] != 0)
                                    {
                                        board[i, j] = board[i, k];
                                        board[i, k] = 0;
                                        combine = true;
                                        break;
                                    }
                                }
                            }
                        }
                        for (int j = 3; j >= 1; j--)
                        {
                            if (board[i, j] == board[i, j - 1] && board[i, j] != 0)
                            {
                                board[i, j] *= 2;
                                combine = true;
                                for (int k = j - 1; k >= 1; k--)
                                {
                                    board[i, k] = board[i, k - 1];
                                }
                                board[i, 0] = 0;
                            }
                        }
                    }
                    break;
                #endregion
                case Keys.Space:
                    b_auto = true;
                    break;
                case Keys.S:
                    b_auto = false;
                    MessageBox.Show("Auto Stop");
                    break;
            }
            if (combine == true && b_paint == true)
            {
                Point nPoint = RndPoint();
                if (nPoint.X == -1 && nPoint.Y == -1)
                {
                    MessageBox.Show("Game Over");
                }
                board[nPoint.Y, nPoint.X] = RndValue();
                PaintRefresh();
            }
        }
        private void Clear()
        {
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    board[i, j] = 0;
                }
            }
            Point initOne = RndPoint();
            board[initOne.Y, initOne.X] = RndValue();
            Point initTwo = RndPoint();
            board[initTwo.Y, initTwo.X] = RndValue();
        }
        private void Form1_Load(object sender, EventArgs e)
        {
            Clear();
            fm = this;
            th.Start();
        }
        private void newGame()
        {
            b_auto = false;
            Thread.Sleep(200);
            Clear();
            PaintRefresh();
        }

        private void Form1_FormClosed(object sender, FormClosedEventArgs e)
        {
            th.Abort();
        }
        private void Form1_MouseClick(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                //MessageBox.Show(e.X.ToString() + ", " + e.Y.ToString());
                if (e.X > 420 && e.X < 546 && e.Y > 139 && e.Y < 176)
                {
                    newGame();
                }
            }
        }
    }
}
